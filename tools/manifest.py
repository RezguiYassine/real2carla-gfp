"""Typed manifest of the page: which accidents it shows and how.

`site_src/accidents.yaml` lists the accidents; `site_src/site.yaml` holds the
paper's title, authors and links. A typo in either fails the build here
instead of producing a broken card.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Collision = Literal["front_to_side", "front_to_rear", "front_to_front", "side_to_side", "other"]
View = Literal["near_real", "top_view", "ego1", "ego2", "lidar1", "lidar2"]
ALL_VIEWS: tuple[View, ...] = ("near_real", "top_view", "ego1", "ego2", "lidar1", "lidar2")

COLLISION_LABEL = {"front_to_side": "front to side", "front_to_rear": "rear-end",
                   "front_to_front": "head-on", "side_to_side": "sideswipe", "other": "other"}


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Accident(_Frozen):
    video: str = Field(pattern=r"^Video \d+$")
    title: str
    collision: Collision
    views: tuple[View, ...] = ALL_VIEWS
    window_s: tuple[float, float] = (-4.0, 3.0)
    download: bool = False
    reason: str | None = None
    # top view with the crash pair's boxes, paths, speeds, gap and time to
    # collision drawn on it (tools/overlay_top_view.py)
    overlay: bool = False

    @property
    def slug(self) -> str:
        return "video-" + self.video.split()[1]


class History(_Frozen):
    video: str = Field(pattern=r"^Video \d+$")
    iterations: tuple[int, ...]
    window_s: tuple[float, float] = (-4.0, 3.0)


class Manifest(_Frozen):
    hero: Accident                  # shown first, all views at once, playing on arrival
    gallery: tuple[Accident, ...]
    failures: tuple[Accident, ...]
    history: History

    @property
    def accidents(self) -> tuple[Accident, ...]:
        return (self.hero,) + self.gallery + self.failures


class Author(_Frozen):
    name: str
    affiliations: tuple[int, ...] = ()


class Link(_Frozen):
    label: str
    url: str | None = None          # None: shown greyed out with `note`
    note: str | None = None


class PriorWork(_Frozen):
    title: str
    venue: str
    year: int


class Site(_Frozen):
    title: str
    short_title: str
    url: str
    authors: tuple[Author, ...]
    affiliations: tuple[str, ...]
    summary: tuple[str, ...]
    abstract: str
    links: tuple[Link, ...]
    bibtex: str
    prior_work: tuple[PriorWork, ...]
    takedown: str


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(yaml.safe_load(path.read_text()))


def load_site(path: Path) -> Site:
    return Site.model_validate(yaml.safe_load(path.read_text()))


def video_number(video: str) -> int:
    m = re.fullmatch(r"Video (\d+)", video)
    if m is None:
        raise ValueError(f"not a video name: {video!r}")
    return int(m.group(1))
