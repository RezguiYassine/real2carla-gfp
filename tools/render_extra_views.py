#!/usr/bin/env python3
"""Render the vehicle-mounted views of a reconstruction: an ego camera and a
semantic LiDAR on each vehicle of the crash pair, plus a CARLA recorder log.

Sim2sim's `render_reconstructions.render_video` runs unchanged, so the scene,
traffic, crash plan and physics are exactly those of the reviewed render. Two
of its module-level names are wrapped for the duration of the run:

- `Encoder`: when the near-real encoder opens, the extra sensors are attached
  to the crash pair; each near-real frame written also writes one frame of
  every extra view, taken at the same simulator frame. All views therefore
  have the same frame count and t = 0 at the same frame as near_real.mp4.
- `teardown`: the extra sensors are stopped and destroyed before the
  renderer's own cleanup.

Run from the Sim2sim folder (its paths are relative):

    cd ../Sim2sim
    devenv shell -- python ../real2carla-gfp/tools/render_extra_views.py \\
        --videos "Video 58,Video 160" --out ../real2carla-gfp/_work/render

Per video it writes <out>/<Video N>/:
    ego1.mp4, ego2.mp4        1280x720 dashcam on each vehicle
    lidar1.mp4, lidar2.mp4    LiDAR sweep seen from behind and above the vehicle
    lidar_contact.npz         sweeps within +/-1.5 s of contact (xyz float16, tag uint8)
    recording.log             CARLA recorder log of the whole run
    near_real.mp4, top_view.mp4, render.json, telemetry.csv   the renderer's own output
    extra_views.json          sensor setup and which vehicle is 1 and 2
"""
from __future__ import annotations

import argparse
import json
import queue
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

FPS = 30.0
EGO_W, EGO_H, EGO_FOV = 1280, 720, 90.0
LIDAR_W, LIDAR_H = 1280, 720
LIDAR_CHANNELS, LIDAR_RANGE_M = 64, 80.0
# 64 channels at one sweep per frame; about 40 000 points per sweep.
LIDAR_PPS = 1_200_000
LIDAR_UPPER_FOV, LIDAR_LOWER_FOV = 15.0, -25.0
# Virtual camera that films the sweep: behind and above the sensor, looking down.
CHASE_BACK_M, CHASE_UP_M, CHASE_PITCH_DEG, CHASE_FOV_DEG = 16.0, 10.0, 26.0, 80.0
CONTACT_WINDOW_S = 1.5
HIDDEN_Z_M = -100.0          # the renderer parks unobserved actors at z = -500
MIN_FREE_GB = 4.0
REVIEWED = Path("results/gfp_v2/sim_videos")     # the reviewed iteration-6 renders (read only)

LIDAR_DTYPE = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("cos", "f4"),
                        ("idx", "u4"), ("tag", "u4")])

# CARLA 0.9.16 semantic tags -> BGR, close to the CityScapes palette CARLA uses.
_TAG_RGB = {
    1: (128, 64, 128), 2: (244, 35, 232), 3: (70, 70, 70), 4: (102, 102, 156),
    5: (190, 153, 153), 6: (153, 153, 153), 7: (250, 170, 30), 8: (220, 220, 0),
    9: (107, 142, 35), 10: (152, 251, 152), 12: (220, 20, 60), 13: (255, 0, 0),
    14: (0, 0, 142), 15: (0, 0, 70), 16: (0, 60, 100), 17: (0, 80, 100),
    18: (0, 0, 230), 19: (119, 11, 32), 20: (110, 190, 160), 21: (170, 120, 50),
    22: (55, 90, 80), 23: (45, 60, 150), 24: (157, 234, 50), 25: (81, 0, 81),
    26: (150, 100, 100), 27: (230, 150, 140), 28: (180, 165, 180),
}
# Road users are drawn brighter than the scene so they stand out in the sweep.
_ROAD_USERS = (12, 13, 14, 15, 16, 17, 18, 19)
TAG_BGR = np.full((256, 3), 90, np.uint8)
for _t, (_r, _g, _b) in _TAG_RGB.items():
    TAG_BGR[_t] = (_b, _g, _r)
for _t in _ROAD_USERS:
    TAG_BGR[_t] = (0, 215, 255) if _t in (14, 15, 16, 17) else (60, 60, 255)


@dataclass(frozen=True, slots=True)
class Mount:
    """Where the sensors sit on one vehicle, from its bounding box."""
    ego_x: float
    ego_z: float
    lidar_z: float
    half_len: float
    half_wid: float


def mount_for(actor) -> Mount:
    bb = actor.bounding_box
    top = bb.location.z + bb.extent.z
    return Mount(ego_x=bb.location.x + 0.4 * bb.extent.x, ego_z=top + 0.25,
                 lidar_z=top + 0.4, half_len=bb.extent.x, half_wid=bb.extent.y)


def _chase_basis():
    th = np.radians(CHASE_PITCH_DEG)
    fwd = np.array([np.cos(th), 0.0, -np.sin(th)])
    right = np.array([0.0, 1.0, 0.0])
    up = np.array([np.sin(th), 0.0, np.cos(th)])
    centre = np.array([-CHASE_BACK_M, 0.0, CHASE_UP_M])
    f_px = (LIDAR_W / 2.0) / np.tan(np.radians(CHASE_FOV_DEG) / 2.0)
    return fwd, right, up, centre, f_px


def _project(pts: np.ndarray):
    """Sensor-frame points (N, 3; x forward, y right, z up) -> pixel u, v, depth."""
    fwd, right, up, centre, f_px = _chase_basis()
    d = pts - centre
    zc, xc, yc = d @ fwd, d @ right, d @ up
    with np.errstate(divide="ignore", invalid="ignore"):
        u = LIDAR_W / 2.0 + f_px * xc / zc
        v = LIDAR_H / 2.0 - f_px * yc / zc
    return u, v, zc


def lidar_frame(raw: np.ndarray, mount: Mount) -> np.ndarray:
    """One sweep drawn from the chase camera, coloured by semantic class, nearer
    points over farther ones, the vehicle itself as a white footprint."""
    img = np.zeros((LIDAR_H, LIDAR_W, 3), np.uint8)
    pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
    u, v, zc = _project(pts)
    ok = (zc > 0.5) & (u >= 0) & (u < LIDAR_W - 1) & (v >= 0) & (v < LIDAR_H - 1)
    order = np.argsort(-zc[ok])                  # far first: near points are written last
    ui, vi = u[ok][order].astype(int), v[ok][order].astype(int)
    rng = np.linalg.norm(pts[ok][order], axis=1)
    shade = np.clip(1.15 - rng / LIDAR_RANGE_M, 0.35, 1.0)[:, None]
    col = (TAG_BGR[raw["tag"][ok][order]] * shade).astype(np.uint8)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        img[vi + dv, ui + du] = col
    gz = -mount.lidar_z
    fp = np.array([[mount.half_len, -mount.half_wid, gz], [mount.half_len, mount.half_wid, gz],
                   [-mount.half_len, mount.half_wid, gz], [-mount.half_len, -mount.half_wid, gz]])
    fu, fv, _ = _project(fp)
    cv2.fillPoly(img, [np.stack([fu, fv], 1).round().astype(np.int32)], (255, 255, 255))
    return img


def caption(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (24, out.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (200, 200, 200), 2, cv2.LINE_AA)
    return out


@dataclass
class Extras:
    """The extra sensors of one video and their encoders."""
    world: object
    out: Path
    tids: tuple[int, ...]
    mirror: bool
    collision_frame_index: int
    sensors: list = field(default_factory=list)
    queues: dict = field(default_factory=dict)
    encoders: dict = field(default_factory=dict)
    actors: dict = field(default_factory=dict)
    mounts: dict = field(default_factory=dict)
    written: int = 0
    contact: dict = field(default_factory=dict)
    closed: bool = False

    def spawn(self, encoder_cls) -> None:
        import carla
        import carla_safe as CSF
        prefix = f"{CSF.RENDER_ROLE_PREFIX}track_"
        by_role = {a.attributes.get("role_name", ""): a for a in self.world.get_actors()}
        lib = self.world.get_blueprint_library()
        for i, tid in enumerate(self.tids, start=1):
            actor = by_role.get(f"{prefix}{tid}")
            if actor is None:
                raise RuntimeError(f"crash-pair actor {tid} not found among the renderer's actors")
            m = mount_for(actor)
            self.actors[i], self.mounts[i] = actor, m
            ego = lib.find("sensor.camera.rgb")
            for k, val in (("image_size_x", EGO_W), ("image_size_y", EGO_H), ("fov", EGO_FOV)):
                ego.set_attribute(k, str(val))
            CSF.tag_blueprint(ego, f"{CSF.RENDER_ROLE_PREFIX}ego_{tid}")
            lid = lib.find("sensor.lidar.ray_cast_semantic")
            for k, val in (("channels", LIDAR_CHANNELS), ("range", LIDAR_RANGE_M),
                           ("points_per_second", LIDAR_PPS), ("rotation_frequency", FPS),
                           ("upper_fov", LIDAR_UPPER_FOV), ("lower_fov", LIDAR_LOWER_FOV)):
                lid.set_attribute(k, str(val))
            CSF.tag_blueprint(lid, f"{CSF.RENDER_ROLE_PREFIX}lidar_{tid}")
            for name, bp, tf in (
                    (f"ego{i}", ego, carla.Transform(carla.Location(x=m.ego_x, z=m.ego_z),
                                                     carla.Rotation(pitch=-5.0))),
                    (f"lidar{i}", lid, carla.Transform(carla.Location(z=m.lidar_z)))):
                time.sleep(CSF.SPAWN_PAUSE_S)
                s = self.world.spawn_actor(bp, tf, attach_to=actor)
                self.sensors.append(s)
                q = queue.Queue()
                self.queues[name] = q
                s.listen(q.put)
                w, h = (EGO_W, EGO_H) if name.startswith("ego") else (LIDAR_W, LIDAR_H)
                self.encoders[name] = encoder_cls(self.out / f"{name}.mp4", w, h)

    def _take(self, name: str, frame: int):
        q = self.queues[name]
        d = q.get(timeout=20.0)
        while d.frame < frame:
            d = q.get(timeout=20.0)
        return d

    def write_frame(self, frame: int) -> None:
        snap = self.world.get_snapshot()
        lo, hi = (self.collision_frame_index - int(CONTACT_WINDOW_S * FPS),
                  self.collision_frame_index + int(CONTACT_WINDOW_S * FPS))
        for i in sorted(self.actors):
            sa = snap.find(self.actors[i].id)
            parked = sa is not None and sa.get_transform().location.z < HIDDEN_Z_M
            img = self._take(f"ego{i}", frame)
            arr = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
            sweep = np.frombuffer(self._take(f"lidar{i}", frame).raw_data, LIDAR_DTYPE)
            lid = lidar_frame(sweep, self.mounts[i])
            if self.mirror:
                # Left-hand traffic was reconstructed as its mirror image; flip back
                # as the renderer does for the near-real view.
                arr, lid = arr[:, ::-1], lid[:, ::-1]
            if parked:
                arr = caption(np.zeros_like(arr), "vehicle not yet observed")
                lid = caption(np.zeros_like(lid), "vehicle not yet observed")
            self.encoders[f"ego{i}"].write(arr)
            self.encoders[f"lidar{i}"].write(lid)
            if lo <= self.written <= hi and not parked:
                self.contact.setdefault(f"lidar{i}", []).append(
                    (self.written, np.stack([sweep["x"], sweep["y"], sweep["z"]], 1).astype(np.float16),
                     sweep["tag"].astype(np.uint8)))
        self.written += 1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for e in self.encoders.values():
            e.close()
        arrays = {}
        for name, sweeps in self.contact.items():
            for k, xyz, tag in sweeps:
                arrays[f"{name}_f{k:04d}_xyz"] = xyz
                arrays[f"{name}_f{k:04d}_tag"] = tag
        if arrays:
            np.savez_compressed(self.out / "lidar_contact.npz", **arrays)

    def destroy(self) -> None:
        for s in self.sensors:
            for fn in (s.stop, s.destroy):
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - cleanup goes on
                    print(f"    extra sensor cleanup: {type(exc).__name__}: {exc}", flush=True)
        self.sensors.clear()


def crash_pair(render_json: Path) -> tuple[int, ...]:
    """The crash pair's vehicles in the renderer's order; a pedestrian gets no
    mounted sensor."""
    actors = json.loads(render_json.read_text())["actors"]
    return tuple(int(k) for k, a in actors.items()
                 if a.get("involved") and not str(a.get("blueprint", "")).startswith("walker"))


def render_one(client, world, video: str, out_root: Path, R) -> dict:
    out = out_root / video
    out.mkdir(parents=True, exist_ok=True)
    reviewed_json = REVIEWED / video / "render.json"
    tids = crash_pair(reviewed_json)
    rec = json.loads((R.RECON / video / "reconstruction.json").read_text())
    reviewed = json.loads(reviewed_json.read_text())
    extras = Extras(world, out, tids, bool(rec.get("mirrored")), int(reviewed["collision_at_frame"]))
    base_encoder, base_teardown = R.Encoder, R.teardown

    class HookedEncoder(base_encoder):
        def __init__(self, path, w, h, fps=FPS):
            super().__init__(path, w, h, fps)
            self.view = Path(path).stem
            if self.view == "near_real":
                extras.spawn(base_encoder)

        def write(self, img):
            super().write(img)
            if self.view == "near_real":
                extras.write_frame(world.get_snapshot().frame)

        def close(self):
            super().close()
            if self.view == "near_real":
                extras.close()

    def hooked_teardown(*a, **kw):
        extras.destroy()
        return base_teardown(*a, **kw)

    R.Encoder, R.teardown = HookedEncoder, hooked_teardown
    log = (out / "recording.log").resolve()
    client.start_recorder(str(log), True)
    try:
        info = R.render_video(client, world, video)
    finally:
        client.stop_recorder()
        R.Encoder, R.teardown = base_encoder, base_teardown
        extras.destroy()
    meta = {"video": video, "frames": extras.written, "collision_at_frame": info["collision_at_frame"],
            "vehicles": {str(i): {"track_id": t, "class": info["actors"][str(t)]["class"]}
                         for i, t in enumerate(tids, start=1)},
            "ego_camera": {"size": [EGO_W, EGO_H], "fov_deg": EGO_FOV, "pitch_deg": -5.0,
                           "mount": "front of the roof, from the bounding box"},
            "lidar": {"type": "sensor.lidar.ray_cast_semantic", "channels": LIDAR_CHANNELS,
                      "range_m": LIDAR_RANGE_M, "points_per_second": LIDAR_PPS,
                      "rotation_frequency_hz": FPS, "fov_deg": [LIDAR_LOWER_FOV, LIDAR_UPPER_FOV],
                      "video_camera": {"back_m": CHASE_BACK_M, "up_m": CHASE_UP_M,
                                       "pitch_deg": CHASE_PITCH_DEG, "fov_deg": CHASE_FOV_DEG}},
            "mirrored": bool(rec.get("mirrored")), "recorder_log": log.name}
    (out / "extra_views.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help='comma-separated, e.g. "Video 58,Video 160"')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    if not Path("render_reconstructions.py").exists():
        raise SystemExit("run from the Sim2sim folder (its paths are relative)")
    free_gb = shutil.disk_usage("/").free / 1e9
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"only {free_gb:.1f} GB free on /; need {MIN_FREE_GB:.0f} GB")
    import carla
    import pandas as pd
    import render_reconstructions as R
    from carla_safe import current_first, healthy, sweep_our_actors

    out_root = args.out.resolve()
    R.OUT = out_root            # the renderer's own files go here too; sim_videos/ is not touched
    names = [v.strip() for v in args.videos.split(",") if v.strip()]
    man = pd.read_csv(R.RECON / "manifest.csv")
    town = man.set_index("video")["map"].to_dict()
    client = carla.Client(args.host, args.port)
    client.set_timeout(180.0)
    world = client.get_world()
    left = sweep_our_actors(world)
    if left:
        print(f"removed {len(left)} actor(s) an earlier run of ours left behind", flush=True)
    order = current_first(world, {town[v] for v in names})
    names.sort(key=lambda v: (order.index(town[v]), v))
    for k, vid in enumerate(names, 1):
        if args.skip_done and (out_root / vid / "extra_views.json").exists():
            continue
        if not healthy(client):
            print(f"[{k}/{len(names)}] CARLA stopped answering; the rest is not rendered", flush=True)
            break
        t0 = time.time()
        try:
            world = R.load_town(client, world, town[vid])
            meta = render_one(client, world, vid, out_root, R)
            time.sleep(1.0)
            print(f"[{k}/{len(names)}] {vid:<11} {town[vid]:<13} {meta['frames']} frames, "
                  f"{len(meta['vehicles'])} vehicle(s)  {time.time() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - one failed video does not stop the run
            print(f"[{k}/{len(names)}] {vid:<11} FAILED {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
