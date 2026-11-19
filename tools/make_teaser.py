#!/usr/bin/env python3
"""Compose the teaser from clips encode_clips.py wrote: for each accident, the
real clip, then the near-real view, the ego camera and the LiDAR of vehicle 1,
each from 3 s before to 1.5 s after first contact, labelled.

    uv run tools/make_teaser.py --videos "Video 108,Video 160"
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FPS = 30
W, H = 960, 540
SEGMENT_S = (-3.0, 1.5)
WINDOW_START_S = -4.0        # encode_clips' default window
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
STEPS = (("real", "real CCTV clip"), ("near_real", "reconstruction, near-real camera"),
         ("ego1", "camera on vehicle 1"), ("lidar1", "LiDAR on vehicle 1"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", default="Video 108,Video 160")
    args = ap.parse_args()
    videos = [v.strip() for v in args.videos.split(",")]
    first = int(round((SEGMENT_S[0] - WINDOW_START_S) * FPS))
    count = int(round((SEGMENT_S[1] - SEGMENT_S[0]) * FPS))
    inputs, chains = [], []
    for vid in videos:
        slug = "video-" + vid.split()[1]
        for view, label in STEPS:
            k = len(inputs) // 2            # inputs holds "-i", path pairs
            inputs += ["-i",str(ROOT / "static/videos" / slug / f"{view}.mp4")]
            chains.append(
                f"[{k}:v]trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS,"
                f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                f"drawtext=fontfile={FONT}:text='{vid}  |  {label}':fontcolor=white:fontsize=26:"
                f"box=1:boxcolor=black@0.6:boxborderw=10:x=20:y=20[s{k}]")
    n = len(chains)
    graph = ";".join(chains) + ";" + "".join(f"[s{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    out = ROOT / "static/videos/teaser.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
                    "-r", str(FPS), "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p",
                    "-g", "30", "-movflags", "+faststart", "-an", str(out)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "3.5", "-i", str(out), "-frames:v", "1",
                    "-q:v", "4", str(out.with_suffix(".jpg"))], check=True)
    print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1e6:.2f} MB, {n} segments")


if __name__ == "__main__":
    main()
