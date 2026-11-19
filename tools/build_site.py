#!/usr/bin/env python3
"""Build the page from the manifest: index.html, plot data and scene downloads.

Reads site_src/{site,accidents}.yaml and, read-only, the Sim2sim results:

    reconstructions/<Video N>/reconstruction.json, vehicle_<id>.csv
    scenes/<Video N>/{scene,collision}.json, agents.csv
    scenarios/<Video N>.json
    sim_videos/<Video N>/render.json, video_review/feedback.json

and writes

    index.html
    static/data/<slug>.json    paths and speeds for the plots
    static/data/<slug>.zip     the scene download (gallery accidents with download: true)

Every entry is checked first: the collision type must equal the scenario's,
and every clip encode_clips.py should have written must exist with the same
frame count. A failed check stops the build.

    uv run tools/build_site.py --sim2sim ../Sim2sim
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from manifest import COLLISION_LABEL, Accident, Manifest, load_manifest, load_site

ROOT = Path(__file__).resolve().parent.parent
TIER_LABEL = {"metric": "metric (≤ 2 m)", "lane": "lane (2–4 m)", "topological": "topological (4–8 m)"}
VIEW_LABEL = {"near_real": "near-real", "top_view": "top view", "ego1": "ego 1", "ego2": "ego 2",
              "lidar1": "LiDAR 1", "lidar2": "LiDAR 2"}
VEHICLE_COLOUR = ("#e0a800", "#c000c0")     # plot colours of vehicle 1 and 2
STATUS_LABEL = {"good": "good", "needs_work": "needs work", None: "not rated"}


@dataclass(frozen=True, slots=True)
class Card:
    """Everything the template needs for one accident."""
    acc: Accident
    road_users: str
    collision_label: str
    tier: str
    uncertainty_m: float
    town: str
    mirrored: bool
    vehicles: tuple[dict, ...]
    real_span: dict
    zip_name: str | None
    zip_mb: float | None


def frame_count(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return int(out.strip())


def check_clips(acc: Accident, videos: Path) -> None:
    d = videos / acc.slug
    names = ["real", *acc.views]
    missing = [n for n in names if not (d / f"{n}.mp4").exists() or not (d / f"{n}.jpg").exists()]
    if missing:
        raise SystemExit(f"{acc.video}: clips missing ({', '.join(missing)}); run tools/encode_clips.py")
    counts = {n: frame_count(d / f"{n}.mp4") for n in names}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"{acc.video}: clips differ in frame count: {counts}")


def involved(render: dict) -> list[tuple[int, str]]:
    return [(int(k), a["class"]) for k, a in render["actors"].items() if a.get("involved")]


def _series(s: pd.Series, digits: int) -> list:
    """A column as a JSON list; a missing value becomes null (JSON has no NaN)."""
    return [None if pd.isna(v) else round(float(v), digits) for v in s]


def plot_data(acc: Accident, res: Path, render: dict) -> dict:
    """Per crash-pair road user: the path the simulation drives (scene agents)
    and the samples GFP measured from the video (vehicle_<id>.csv)."""
    agents = pd.read_csv(res / "scenes" / acc.video / "agents.csv")
    out = []
    for k, (tid, cls) in enumerate(involved(render)):
        a = agents[agents.track_id == tid].sort_values("t")
        meas_file = res / "reconstructions" / acc.video / f"vehicle_{tid}.csv"
        m = pd.read_csv(meas_file) if meas_file.exists() else pd.DataFrame(columns=["t_s", "x_m", "y_m", "speed_ms"])
        out.append({
            "id": tid, "class": cls, "colour": VEHICLE_COLOUR[k % 2] if cls != "person" else "#3a7bd5",
            "scene": {"t": _series(a.t, 3), "x": _series(a.x, 2), "y": _series(a.y, 2),
                      "speed": _series(a.speed, 2)},
            "measured": {"t": _series(m.t_s, 3), "x": _series(m.x_m, 2), "y": _series(m.y_m, 2),
                         "speed": _series(m.speed_ms, 2)},
        })
    return {"video": acc.video, "window_s": list(acc.window_s), "road_users": out}


ZIP_README = """# {video}: Real2CARLA-GFP scene

A reconstruction of one real accident from the CADP videos, as Real2CARLA-GFP
produced it for the paper (review iteration 6, rated {status}). Time 0 is the
first contact. CARLA version: 0.9.16. Map: {town}.{mirror}

## Files

- `scenario.json`: the scenario a person confirmed for this video (schema 2):
  road layout, how each road user arrives and moves, the collision (type,
  striker, struck side, where), traffic. Values marked `unknown` were left to
  the pipeline's defaults.
- `reconstruction.json`: what GFP estimated from one frame of the video.
  `pose`: pitch_deg, roll_deg, fov_deg, height_m of the real camera.
  `capability_tier` and `position_uncertainty_m`: GFP's own estimate of its
  position error. `collision.frame`: the contact frame in the real clip.
  `camera`: the near-real CARLA camera.
- `vehicle_<id>.csv`: the trajectory GFP measured for each crash-pair road
  user. Columns: `t_s` (s, 0 = contact), `frame` (real-video frame),
  `x_m, y_m, z_m` (CARLA world metres), `heading_deg`, `speed_ms` (m/s),
  `s_to_collision_m` (distance along the path to the contact point).
- `agents.csv`: every road user the simulation drives over [-8 s, +6 s],
  including traffic and the history invented before the vehicle was seen.
  Columns: `track_id, t, x, y, z, yaw, speed, source, class, involved,
  placement, confidence, ...`.
- `scene.json`: how the scene was built (map, window, agents by class, what
  was placed, cleaned or dropped).
- `collision.json`: the collision solver's plan for the crash pair (timing,
  sideways and speed changes) and its verification under CARLA physics.
- `recording.log`: CARLA recorder log of the rendered run. Replay it in CARLA
  0.9.16 with your own sensors:

      client.load_world("{town}")
      client.replay_file("/absolute/path/recording.log", 0, 0, 0)

  Scale caveat: absolute distances and speeds depend on the estimated camera
  height and are uncertain by tens of percent; the lane, manoeuvre, event
  order and collision geometry are the reliable parts.

License of the scene files: CC BY 4.0. The real video is from the CADP
dataset and is not included.
"""


def build_zip(acc: Accident, res: Path, render_dir: Path, out: Path, status: str, town: str,
              mirrored: bool) -> Path:
    z = out / f"{acc.slug}.zip"
    rec_dir = res / "reconstructions" / acc.video
    files = {
        "scenario.json": res / "scenarios" / f"{acc.video}.json",
        "reconstruction.json": rec_dir / "reconstruction.json",
        "agents.csv": res / "scenes" / acc.video / "agents.csv",
        "scene.json": res / "scenes" / acc.video / "scene.json",
        "collision.json": res / "scenes" / acc.video / "collision.json",
        "recording.log": render_dir / acc.video / "recording.log",
        **{f.name: f for f in sorted(rec_dir.glob("vehicle_*.csv"))},
    }
    missing = [n for n, f in files.items() if not f.exists()]
    if missing:
        raise SystemExit(f"{acc.video}: download files missing: {', '.join(missing)}")
    mirror = ("\nLeft-hand traffic: the scene is reconstructed as its right-hand mirror image "
              "(the videos on the page are flipped back)." if mirrored else "")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        folder = acc.slug
        for name, f in files.items():
            zf.write(f, f"{folder}/{name}")
        # dated like its newest input, so an unchanged scene rebuilds byte for byte
        newest = time.localtime(max(f.stat().st_mtime for f in files.values()))[:6]
        zf.writestr(zipfile.ZipInfo(f"{folder}/README.md", date_time=newest),
                    ZIP_README.format(video=acc.video, status=status, town=town, mirror=mirror),
                    compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return z


def latest_status(feedback: dict, video: str, iteration: int) -> str | None:
    return (feedback.get(video, {}).get(str(iteration)) or {}).get("status")


def make_card(acc: Accident, res: Path, render_dir: Path, data: Path, spans: dict, feedback: dict,
              current: int) -> Card:
    scenario = json.loads((res / "scenarios" / f"{acc.video}.json").read_text())
    if scenario["collision"]["type"] != acc.collision:
        raise SystemExit(f"{acc.video}: manifest says {acc.collision}, scenario says "
                         f"{scenario['collision']['type']}")
    rec = json.loads((res / "reconstructions" / acc.video / "reconstruction.json").read_text())
    render = json.loads((res / "sim_videos" / acc.video / "render.json").read_text())
    status = latest_status(feedback, acc.video, current)
    want = "needs_work" if acc.reason else "good"
    if status != want:
        raise SystemExit(f"{acc.video}: iteration {current} status is {status}, the manifest lists it as {want}")
    town = rec["map"].removesuffix("_Opt")
    (data / f"{acc.slug}.json").write_text(json.dumps(plot_data(acc, res, render), separators=(",", ":"), allow_nan=False))
    z = build_zip(acc, res, render_dir, data, STATUS_LABEL[status], town, bool(rec.get("mirrored"))) \
        if acc.download else None
    users = involved(render)
    return Card(acc=acc, road_users=" – ".join(c for _, c in users),
                collision_label=COLLISION_LABEL[acc.collision],
                tier=TIER_LABEL.get(rec.get("capability_tier"), str(rec.get("capability_tier"))),
                uncertainty_m=round(float(rec.get("position_uncertainty_m") or 0.0), 1), town=town,
                mirrored=bool(rec.get("mirrored")),
                vehicles=tuple({"n": k + 1, "id": tid, "class": cls}
                               for k, (tid, cls) in enumerate(u for u in users if u[1] != "person")),
                real_span=spans[acc.video], zip_name=z.name if z else None,
                zip_mb=round(z.stat().st_size / 1e6, 1) if z else None)


def history_cards(man: Manifest, feedback: dict) -> list[dict]:
    return [{"it": it, "status": STATUS_LABEL[latest_status(feedback, man.history.video, it)],
             "status_key": latest_status(feedback, man.history.video, it) or "none"}
            for it in man.history.iterations]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim2sim", type=Path, default=ROOT.parent / "Sim2sim")
    ap.add_argument("--render", type=Path, default=ROOT / "_work/render")
    args = ap.parse_args()
    res = args.sim2sim / "results/gfp_v2"
    site = load_site(ROOT / "site_src/site.yaml")
    man = load_manifest(ROOT / "site_src/accidents.yaml")
    spans = json.loads((ROOT / "site_src/real_spans.json").read_text())
    feedback = json.loads((res / "video_review/feedback.json").read_text())
    current = int((res / "sim_videos/ITERATION").read_text())
    videos, data = ROOT / "static/videos", ROOT / "static/data"
    data.mkdir(parents=True, exist_ok=True)
    for acc in man.accidents:
        check_clips(acc, videos)
    hero = make_card(man.hero, res, args.render, data, spans, feedback, current)
    gallery = [make_card(a, res, args.render, data, spans, feedback, current) for a in man.gallery]
    failures = [make_card(a, res, args.render, data, spans, feedback, current) for a in man.failures]
    hist_dir = videos / "history"
    for it in man.history.iterations:
        if not (hist_dir / f"it{it}.mp4").exists():
            raise SystemExit(f"history clip it{it}.mp4 missing; run tools/encode_clips.py")
    history_slug = next(a.slug for a in man.accidents if a.video == man.history.video)
    env = Environment(loader=FileSystemLoader(ROOT / "site_src"), autoescape=True,
                      undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    html = env.get_template("index.html.j2").render(
        site=site, hero=hero, gallery=gallery, failures=failures, history=history_cards(man, feedback),
        history_video=man.history.video, history_slug=history_slug, history_window=man.history.window_s,
        history_span=spans[man.history.video], view_label=VIEW_LABEL,
        collisions=sorted({c.acc.collision for c in gallery}, key=list(COLLISION_LABEL).index),
        collision_label=COLLISION_LABEL)
    (ROOT / "index.html").write_text(html)
    total = sum(f.stat().st_size for f in (ROOT / "static").rglob("*") if f.is_file())
    print(f"index.html: {len(gallery)} gallery, {len(failures)} failures; static/ {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
