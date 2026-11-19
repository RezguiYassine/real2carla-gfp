#!/usr/bin/env python3
"""Draw the crash pair on the top view: 3-D boxes, paths, speeds, the gap
between the two bodies and the time to collision.

Everything comes from the render itself, so the drawing sits exactly on the
rendered cars:
    top_view.mp4, render.json    frames and the top camera's pose and field of view
    telemetry.csv                each actor's pose and speed at every rendered frame
    extra_views.json             which car is vehicle 1 and 2, and their bounding boxes

    uv run tools/overlay_top_view.py "_work/render/Video 201" --aspect 4:3

writes top_view_overlay.mp4 next to the inputs, with as many frames as top_view.mp4.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
from PIL import Image, ImageDraw, ImageFont

FPS = 30.0
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# Vehicle 1 and 2, the same colours as the captions on the site (RGB).
COLOURS = ((41, 182, 246), (255, 87, 51))
PANEL_RGBA = (32, 34, 38, 200)
TTC_WARN_S, TTC_ALERT_S = 3.0, 1.5
MIN_CLOSING_MS = 0.3          # below this the pair is not closing in: no time to collision
TTC_MAX_S = 10.0
CONTACT_GAP_M = 0.05
RATE_HALF_WINDOW = 3          # frames each side for the closing speed (0.2 s in all)

# The 12 edges of a box whose corners are numbered by the bits (x, y, z).
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
CORNERS = np.array([[1 if i & 4 else -1, 1 if i & 2 else -1, 1 if i & 1 else -1] for i in range(8)], float)


def rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """CARLA (Unreal) rotation in degrees -> matrix from the local frame (x forward,
    y right, z up) to the world frame; carla.Transform.get_matrix uses the same terms."""
    p, y, r = np.radians([pitch, yaw, roll])
    cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                     [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                     [sp, -cp * sr, cp * cr]])


@dataclass(frozen=True, slots=True)
class Camera:
    pos: np.ndarray
    rot: np.ndarray
    f_px: float
    width: int
    height: int
    mirror: bool

    @classmethod
    def from_render(cls, cam: dict, mirror: bool) -> "Camera":
        p = cam["pose"]
        w, h = int(cam["width"]), int(cam["height"])
        return cls(np.array([p["x"], p["y"], p["z"]]), rotation(p["pitch"], p["yaw"], p["roll"]),
                   (w / 2.0) / np.tan(np.radians(p["fov"]) / 2.0), w, h, mirror)

    def project(self, world: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> pixels (..., 2); the camera looks along its x."""
        local = (world - self.pos) @ self.rot          # world -> camera axes
        u = self.width / 2.0 + self.f_px * local[..., 1] / local[..., 0]
        v = self.height / 2.0 - self.f_px * local[..., 2] / local[..., 0]
        if self.mirror:
            u = self.width - 1 - u
        return np.stack([u, v], axis=-1)


@dataclass(frozen=True, slots=True)
class Vehicle:
    number: int
    track_id: int
    colour: tuple[int, int, int]
    centre: np.ndarray            # bounding-box centre in the vehicle frame
    extent: np.ndarray            # half sizes
    pose: pd.DataFrame            # per frame: x y z yaw pitch roll speed_ms visible

    def corners(self, frame: int) -> np.ndarray:
        r = self.pose.loc[frame]
        local = self.centre + CORNERS * self.extent
        return local @ rotation(r.pitch, r.yaw, r.roll).T + np.array([r.x, r.y, r.z])

    def ground_path(self) -> np.ndarray:
        return self.pose[["x", "y", "z"]].to_numpy()

    def footprints(self) -> np.ndarray:
        """Each frame's footprint as a shapely polygon (vectorised over frames)."""
        yaw = np.radians(self.pose.yaw.to_numpy())
        ex, ey = self.extent[:2]
        local = np.array([[ex, ey], [ex, -ey], [-ex, -ey], [-ex, ey]]) + self.centre[:2]
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        xs = self.pose.x.to_numpy()[:, None] + c * local[:, 0] - s * local[:, 1]
        ys = self.pose.y.to_numpy()[:, None] + s * local[:, 0] + c * local[:, 1]
        return shapely.polygons(np.stack([xs, ys], axis=-1))


def load_vehicles(render_dir: Path) -> list[Vehicle]:
    meta = json.loads((render_dir / "extra_views.json").read_text())
    tel = pd.read_csv(render_dir / "telemetry.csv")
    out = []
    for key, v in sorted(meta["vehicles"].items(), key=lambda kv: int(kv[0])):
        n = int(key)
        pose = tel[tel.id == v["track_id"]].set_index("frame").sort_index()
        out.append(Vehicle(n, int(v["track_id"]), COLOURS[n - 1], np.array(v["box"]["centre_m"]),
                           np.array(v["box"]["half_extent_m"]),
                           pose[["x", "y", "z", "yaw", "pitch", "roll", "speed_ms", "visible"]]))
    return out


def gap_and_ttc(a: Vehicle, b: Vehicle) -> tuple[np.ndarray, np.ndarray]:
    """Gap between the two footprints (m) and time to collision at the current
    closing speed (s; NaN when they are not closing in)."""
    gap = shapely.distance(a.footprints(), b.footprints())
    k = RATE_HALF_WINDOW
    padded = np.pad(gap, k, mode="edge")
    closing = -(padded[2 * k:] - padded[:-2 * k]) / (2 * k / FPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(closing > MIN_CLOSING_MS, gap / closing, np.nan)
    return gap, ttc


def visible(v: Vehicle, frame: int) -> bool:
    r = v.pose.loc[frame]
    return bool(r.visible) and r.z > -100.0       # the renderer parks unobserved actors at z = -500


def draw_lines(img: np.ndarray, cam: Camera, vehicles: list[Vehicle], frame: int, y0: int) -> np.ndarray:
    """Paths (whole path faint, travelled part solid) and boxes, with OpenCV."""
    faint = img.copy()
    for v in vehicles:
        bgr = v.colour[::-1]
        path = cam.project(v.ground_path()) - (0, y0)
        ok = v.pose.visible.to_numpy().astype(bool) & (v.pose.z.to_numpy() > -100.0)
        idx = np.flatnonzero(ok)
        if idx.size < 2:
            continue
        pts = path[idx].round().astype(np.int32)
        cv2.polylines(faint, [pts], False, bgr, 2, cv2.LINE_AA)
    out = cv2.addWeighted(faint, 0.45, img, 0.55, 0)
    for v in vehicles:
        bgr = v.colour[::-1]
        path = cam.project(v.ground_path()) - (0, y0)
        ok = v.pose.visible.to_numpy().astype(bool) & (v.pose.z.to_numpy() > -100.0)
        idx = np.flatnonzero(ok & (v.pose.index.to_numpy() <= frame))
        if idx.size >= 2:
            cv2.polylines(out, [path[idx].round().astype(np.int32)], False, bgr, 4, cv2.LINE_AA)
        if visible(v, frame):
            box = (cam.project(v.corners(frame)) - (0, y0)).round().astype(int)
            for i, j in EDGES:
                cv2.line(out, tuple(box[i]), tuple(box[j]), bgr, 2, cv2.LINE_AA)
    return out


@dataclass(frozen=True, slots=True)
class Fonts:
    label: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    big: ImageFont.FreeTypeFont

    @classmethod
    def for_width(cls, w: int) -> "Fonts":
        s = w / 1024
        return cls(ImageFont.truetype(FONT_BOLD, round(19 * s)), ImageFont.truetype(FONT, round(17 * s)),
                   ImageFont.truetype(FONT_BOLD, round(26 * s)))


def ttc_text(gap: float, ttc: float) -> tuple[str, tuple[int, int, int]]:
    if gap <= CONTACT_GAP_M:
        return "contact", (255, 82, 82)
    if np.isnan(ttc):
        return "—", (230, 230, 230)
    colour = (255, 82, 82) if ttc < TTC_ALERT_S else (255, 193, 7) if ttc < TTC_WARN_S else (230, 230, 230)
    return (f"{ttc:.1f} s" if ttc < TTC_MAX_S else f"> {TTC_MAX_S:.0f} s"), colour


def edge_arrow(cx: float, cy: float, w: int, h: int) -> str:
    """An arrow towards a vehicle outside the frame, or nothing when it is inside."""
    dx, dy = (cx < 0) * -1 + (cx > w), (cy < 0) * -1 + (cy > h)
    if not (dx or dy):
        return ""
    if abs(dy) >= abs(dx):
        return "▲ " if dy < 0 else "▼ "
    return "◀ " if dx < 0 else "▶ "


def draw_text(img: np.ndarray, cam: Camera, vehicles: list[Vehicle], frame: int, y0: int,
              t: float, gap: float, ttc: float, fonts: Fonts) -> np.ndarray:
    """Labels, legend, clock and the gap / time-to-collision panel, with Pillow."""
    h, w = img.shape[:2]
    pad = round(10 * w / 1024)
    pil = Image.fromarray(img[:, :, ::-1]).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    def box_text(xy, text, font, fill=(255, 255, 255)):
        x0, y0_, x1, y1 = d.textbbox(xy, text, font=font)
        d.rounded_rectangle((x0 - pad, y0_ - pad * 0.6, x1 + pad, y1 + pad * 0.6), radius=pad * 0.8, fill=PANEL_RGBA)
        d.text(xy, text, font=font, fill=fill)

    shown = [v for v in vehicles if visible(v, frame)]
    boxes = {v.number: cam.project(v.corners(frame)) - (0, y0) for v in shown}
    # labels keep off every box and off each other
    taken = [(b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()) for b in boxes.values()]
    for v in shown:
        box = boxes[v.number]
        cx, cy = box[:, 0].mean(), box[:, 1].mean()
        text = (f"{edge_arrow(cx, cy, w, h)}Vehicle {v.number}  |  "
                f"{v.pose.loc[frame].speed_ms * 3.6:.1f} km/h")
        tw, th = d.textlength(text, font=fonts.label), fonts.label.size
        x0b, y0b, x1b, y1b = box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()
        candidates = [(cx - tw / 2, y0b - 2.4 * pad - th), (cx - tw / 2, y1b + 2.0 * pad),
                      (x1b + 2 * pad, cy - th / 2), (x0b - 2 * pad - tw, cy - th / 2)]
        options = []
        for x, y in candidates:
            x = float(np.clip(x, 2 * pad, w - tw - 2 * pad))
            y = float(np.clip(y, 2 * pad, h - th - 2 * pad))
            options.append((x, y, (x - pad, y - pad, x + tw + pad, y + th + pad)))
        free = [o for o in options
                if not any(r[0] < o[2][2] and o[2][0] < r[2] and r[1] < o[2][3] and o[2][1] < r[3] for r in taken)]
        x, y, rect = (free or options)[0]
        taken.append(rect)
        box_text((x, y), text, fonts.label)

    # legend, top left
    lx, ly = 2 * pad, 2 * pad
    rows = [(v.colour, f"Vehicle {v.number}: car {v.track_id}") for v in vehicles]
    lw = max(d.textlength(r[1], font=fonts.small) for r in rows) + 4.5 * pad
    d.rounded_rectangle((lx - pad, ly - pad, lx + lw, ly + len(rows) * 2.6 * pad + 0.4 * pad),
                        radius=pad, fill=PANEL_RGBA)
    for i, (colour, text) in enumerate(rows):
        cy = ly + i * 2.6 * pad + fonts.small.size / 2 + 2
        d.ellipse((lx, cy - 0.7 * pad, lx + 1.4 * pad, cy + 0.7 * pad), fill=colour)
        d.text((lx + 2.4 * pad, ly + i * 2.6 * pad), text, font=fonts.small, fill=(255, 255, 255))

    # clock, top right
    clock = f"t = {'−' if t < -0.05 else '+' if t > 0.05 else ''}{abs(t):.1f} s"
    box_text((w - d.textlength(clock, font=fonts.label) - 2 * pad, 2 * pad), clock, fonts.label)

    # gap and time to collision, bottom right
    a, b = vehicles[0], vehicles[1]
    after = t >= 0 and gap > CONTACT_GAP_M
    title = f"VEHICLE {a.number} ↔ {b.number}"
    longest = f"{title}  ·  after contact"
    title = longest if after else title
    # one width for the whole clip, wide enough for the longest title
    pw, ph = max(26 * pad, d.textlength(longest, font=fonts.label) + 3 * pad), 9.4 * pad
    px, py = w - pw - 2 * pad, h - ph - 2 * pad
    d.rounded_rectangle((px, py, px + pw, py + ph), radius=pad, fill=PANEL_RGBA)
    d.text((px + 1.5 * pad, py + 1.2 * pad), title, font=fonts.label, fill=(255, 255, 255))
    d.text((px + 1.5 * pad, py + 4 * pad), "Gap", font=fonts.small, fill=(190, 190, 190))
    d.text((px + 13.5 * pad, py + 4 * pad), "TTC", font=fonts.small, fill=(190, 190, 190))
    d.text((px + 1.5 * pad, py + 6 * pad), f"{max(gap, 0.0):.1f} m", font=fonts.big, fill=(255, 255, 255))
    text, colour = ("—", (190, 190, 190)) if after else ttc_text(gap, ttc)
    d.text((px + 13.5 * pad, py + 6 * pad), text, font=fonts.big, fill=colour)

    return np.asarray(Image.alpha_composite(pil, layer).convert("RGB"))[:, :, ::-1].copy()


def crop_rows(cam: Camera, aspect: tuple[int, int] | None) -> tuple[int, int]:
    """Rows to keep for the requested aspect, centred (the top camera is centred on contact)."""
    if aspect is None:
        return 0, cam.height
    keep = min(cam.height, round(cam.width * aspect[1] / aspect[0] / 2) * 2)
    y0 = (cam.height - keep) // 2
    return y0, y0 + keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("render_dir", type=Path)
    ap.add_argument("--aspect", default=None, help='crop to this aspect, e.g. "4:3"')
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rd = args.render_dir
    render = json.loads((rd / "render.json").read_text())
    cam = Camera.from_render(render["cameras"]["top_view"], bool(render.get("mirrored")))
    vehicles = load_vehicles(rd)
    if len(vehicles) != 2:
        raise SystemExit(f"{rd}: the overlay needs a crash pair of two vehicles, found {len(vehicles)}")
    gap, ttc = gap_and_ttc(*vehicles)
    cf = int(render["collision_at_frame"])
    aspect = tuple(map(int, args.aspect.split(":"))) if args.aspect else None
    y0, y1 = crop_rows(cam, aspect)
    fonts = Fonts.for_width(cam.width)
    out = args.out or rd / "top_view_overlay.mp4"
    cap = cv2.VideoCapture(str(rd / "top_view.mp4"))
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{cam.width}x{y1 - y0}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                            "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", str(out)],
                           stdin=subprocess.PIPE)
    frame = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = draw_lines(img[y0:y1], cam, vehicles, frame, y0)
        img = draw_text(img, cam, vehicles, frame, y0, (frame - cf) / FPS, float(gap[frame]),
                        float(ttc[frame]), fonts)
        enc.stdin.write(img.tobytes())
        frame += 1
    cap.release()
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    if frame != len(gap):
        raise SystemExit(f"{rd}: {frame} video frames but {len(gap)} telemetry frames")
    i0 = cf - int(4 * FPS)
    print(f"{out}: {frame} frames, {cam.width}x{y1 - y0}; gap at contact {gap[cf]:.2f} m, "
          f"TTC 1 s before contact {ttc[cf - int(FPS)]:.2f} s, 4 s before {ttc[i0]:.2f} s")


if __name__ == "__main__":
    main()
