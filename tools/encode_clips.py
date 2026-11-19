#!/usr/bin/env python3
"""Cut every clip the page plays to the same window around first contact.

For each accident in site_src/accidents.yaml this writes
static/videos/<slug>/{real,near_real,top_view,ego1,ego2,lidar1,lidar2}.mp4
and a poster frame per clip, and for the iteration history
static/videos/history/it<N>.mp4.

All clips of one accident share the frame count: the window at 30 fps, t = 0
at the contact. The real clip's contact is `collision.frame` of
reconstruction.json over the file's frame rate, as in Sim2sim's review tool.
Where the real recording starts after or ends before the window, its first or
last frame is held and labelled, so the real clip keeps the same length.

    uv run tools/encode_clips.py --sim2sim ../Sim2sim --render _work/render
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from manifest import Accident, load_manifest

FPS = 30
MAX_W = 960                  # the page shows each clip at about half its width
CRF = {"lidar": 30}          # LiDAR sweeps are dense dots; a higher CRF keeps them under 3 MB
DEFAULT_CRF = 28
REAL_DIRS = (Path("/media/safe/secondary disk/Jonas/Input2"), Path("/media/safe/secondary disk/Jonas/Input"))
FPS_TABLE = Path("results/gfp_v2/video_fps.csv")      # in Sim2sim: measured frame rate and original file
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Job:
    src: Path
    dst: Path
    vf: str
    seek: tuple[float, float] | None     # input -ss / -t for the real clip
    frames: int
    crf: int


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=avg_frame_rate,width,height:format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    num, den = map(int, d["streams"][0]["avg_frame_rate"].split("/"))
    return {"fps": num / den, "duration": float(d["format"]["duration"]),
            "width": d["streams"][0]["width"], "height": d["streams"][0]["height"]}


def n_frames(window: tuple[float, float]) -> int:
    return int(round((window[1] - window[0]) * FPS)) + 1


def scale(w: int) -> str:
    return f"scale='min({MAX_W},iw)':-2" if w > MAX_W else "scale=trunc(iw/2)*2:trunc(ih/2)*2"


def real_job(src: Path, contact_frame: float, window: tuple[float, float], dst: Path) -> tuple[Job, dict]:
    p = probe(src)
    tc = contact_frame / p["fps"]
    start, end = tc + window[0], tc + window[1]
    a, b = max(0.0, start), min(p["duration"], end)
    pad_before, pad_after = a - start, end - b
    hold = f"between(t,0,{pad_before:.3f})+gte(t,{pad_before + (b - a):.3f})"
    vf = (f"fps={FPS},{scale(p['width'])},"
          f"tpad=start_duration={pad_before:.3f}:stop_duration={pad_after + 1:.3f}:start_mode=clone:stop_mode=clone,"
          f"drawtext=fontfile={FONT}:text='no real footage':fontcolor=white:fontsize=h/14:"
          f"box=1:boxcolor=black@0.6:boxborderw=8:x=(w-tw)/2:y=h-th-h/12:enable='{hold}'")
    span = {"source_fps": round(p["fps"], 3), "contact_s_in_source": round(tc, 3),
            "real_from_s": round(a - tc, 3), "real_to_s": round(b - tc, 3)}
    return Job(src, dst, vf, (a, b - a), n_frames(window), DEFAULT_CRF), span


def sim_job(src: Path, contact_frame: int, window: tuple[float, float], dst: Path, kind: str) -> Job:
    first = contact_frame + int(round(window[0] * FPS))
    if first < 0:
        raise ValueError(f"{src}: the render starts after the window")
    w = probe(src)["width"]
    vf = f"trim=start_frame={first},setpts=PTS-STARTPTS,{scale(w)}"
    return Job(src, dst, vf, None, n_frames(window), CRF.get(kind, DEFAULT_CRF))


def run(job: Job) -> Path:
    job.dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if job.seek is not None:
        cmd += ["-ss", f"{job.seek[0]:.3f}", "-t", f"{job.seek[1] + 0.5:.3f}"]
    cmd += ["-i", str(job.src), "-an", "-vf", job.vf, "-frames:v", str(job.frames), "-r", str(FPS),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(job.crf), "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-g", "15", "-movflags", "+faststart", str(job.dst)]
    subprocess.run(cmd, check=True)
    poster = job.dst.with_suffix(".jpg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(job.dst), "-frames:v", "1",
                    "-vf", "scale='min(640,iw)':-2", "-q:v", "5", str(poster)], check=True)
    return job.dst


def real_source(video: str, sim2sim: Path) -> Path:
    """The original clip. A sub-clip (Video 201) plays from its parent file, as
    in Sim2sim's review tool; the frame-rate table names the parent."""
    for d in REAL_DIRS:
        f = d / f"{video}.mp4"
        if f.exists():
            return f
    table = pd.read_csv(sim2sim / FPS_TABLE).fillna("").set_index("video")
    if video in table.index and table.at[video, "original"] and Path(table.at[video, "original"]).exists():
        return Path(table.at[video, "original"])
    raise FileNotFoundError(f"{video}: no original clip in {', '.join(map(str, REAL_DIRS))} or {FPS_TABLE}")


def accident_jobs(acc: Accident, sim2sim: Path, render: Path, out: Path) -> tuple[list[Job], dict]:
    res = sim2sim / "results/gfp_v2"
    rec = json.loads((res / "reconstructions" / acc.video / "reconstruction.json").read_text())
    reviewed = json.loads((res / "sim_videos" / acc.video / "render.json").read_text())
    extra = render / acc.video
    extra_meta = json.loads((extra / "extra_views.json").read_text())
    if extra_meta["collision_at_frame"] != reviewed["collision_at_frame"]:
        raise ValueError(f"{acc.video}: extra views and the reviewed render disagree on the contact frame")
    dst = out / acc.slug
    real, span = real_job(real_source(acc.video, sim2sim), float(rec["collision"]["frame"]), acc.window_s,
                          dst / "real.mp4")
    jobs = [real]
    for view in acc.views:
        src = (res / "sim_videos" / acc.video if view in ("near_real", "top_view") else extra) / f"{view}.mp4"
        if view == "top_view" and acc.overlay:
            src = extra / "top_view_overlay.mp4"
            if not src.exists():
                raise FileNotFoundError(f"{src} missing; run tools/overlay_top_view.py \"{extra}\" --aspect 4:3")
        jobs.append(sim_job(src, int(reviewed["collision_at_frame"]), acc.window_s, dst / f"{view}.mp4",
                            "lidar" if view.startswith("lidar") else view))
    return jobs, span


def history_jobs(video: str, iterations: tuple[int, ...], window: tuple[float, float], sim2sim: Path,
                 out: Path) -> list[Job]:
    res = sim2sim / "results/gfp_v2"
    current = int((res / "sim_videos/ITERATION").read_text())   # earlier ones are archived as sim_videos_iterationN
    jobs = []
    for it in iterations:
        d = res / ("sim_videos" if it == current else f"sim_videos_iteration{it}") / video
        cf = int(json.loads((d / "render.json").read_text())["collision_at_frame"])
        jobs.append(sim_job(d / "near_real.mp4", cf, window, out / "history" / f"it{it}.mp4", "near_real"))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim2sim", type=Path, default=ROOT.parent / "Sim2sim")
    ap.add_argument("--render", type=Path, default=ROOT / "_work/render")
    ap.add_argument("--only", default="", help='comma-separated videos, e.g. "Video 58"')
    args = ap.parse_args()
    man = load_manifest(ROOT / "site_src/accidents.yaml")
    out = ROOT / "static/videos"
    only = {v.strip() for v in args.only.split(",") if v.strip()}
    jobs, spans = [], {}
    for acc in man.accidents:
        if only and acc.video not in only:
            continue
        js, span = accident_jobs(acc, args.sim2sim, args.render, out)
        jobs += js
        spans[acc.video] = span
    if not only or man.history.video in only:
        jobs += history_jobs(man.history.video, man.history.iterations, man.history.window_s, args.sim2sim, out)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for p in pool.map(run, jobs):
            print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.2f} MB", flush=True)
    span_file = ROOT / "site_src/real_spans.json"
    old = json.loads(span_file.read_text()) if span_file.exists() else {}
    span_file.write_text(json.dumps({**old, **spans}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
