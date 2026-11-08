"""The object catalog: which objects exist, their per-object settings, and which
one is active."""

from __future__ import annotations

import random
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .params import ROOT

CATALOG_PATH = ROOT / "catalog.toml"


@dataclass
class ObjectSpec:
    name: str
    group: str = "Object"
    rotation: float = 30.0          # +/- degrees searched; 360 = any angle
    threshold_offset: float = 0.0   # set by tools/calibrate.py
    extra: dict = field(default_factory=dict)

    @property
    def full_rotation(self) -> bool:
        return self.rotation >= 180


@dataclass
class CatalogFile:
    objects: list[ObjectSpec]
    reserves: list[str]


def load_catalog(path: str | Path | None = None) -> CatalogFile:
    path = Path(path) if path else CATALOG_PATH
    with open(path, "rb") as f:
        data = tomllib.load(f)
    objects = []
    for o in data.get("object", []):
        o = dict(o)
        spec = ObjectSpec(
            name=o.pop("name"),
            group=o.pop("group", "Object"),
            rotation=float(o.pop("rotation", 30.0)),
            threshold_offset=float(o.pop("threshold_offset", 0.0)),
        )
        spec.extra = o
        objects.append(spec)
    return CatalogFile(objects=objects, reserves=list(data.get("reserves", [])))


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(round(v, 4)) if isinstance(v, float) else str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v)} to TOML")


def save_catalog(cat: CatalogFile, path: str | Path | None = None) -> None:
    path = Path(path) if path else CATALOG_PATH
    lines = [
        "# Objects the piece can see. Names are Quick, Draw! category names.",
        "# rotation: +/- degrees searched (360 = any angle).",
        "# threshold_offset: written by tools/calibrate.py so every object appears at a similar rate.",
        "",
        "reserves = " + _toml_value(cat.reserves),
        "",
    ]
    for o in cat.objects:
        lines.append("[[object]]")
        lines.append(f"name = {_toml_value(o.name)}")
        lines.append(f"group = {_toml_value(o.group)}")
        lines.append(f"rotation = {_toml_value(int(o.rotation) if float(o.rotation).is_integer() else o.rotation)}")
        lines.append(f"threshold_offset = {_toml_value(float(o.threshold_offset))}")
        for k, v in o.extra.items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    Path(path).write_text("\n".join(lines))


class Catalog:
    """Tracks the active object and handles stepping and shuffle."""

    def __init__(self, objects: list[ObjectSpec], start: str | None = None,
                 shuffle_minutes: float = 0.0):
        if not objects:
            raise ValueError("catalog is empty")
        self.objects = objects
        self.index = 0
        if start:
            names = [o.name for o in objects]
            if start in names:
                self.index = names.index(start)
        self.shuffle_minutes = shuffle_minutes
        self.shuffle_on = shuffle_minutes > 0
        self._last_switch = time.monotonic()

    @property
    def active(self) -> ObjectSpec:
        return self.objects[self.index]

    def names(self) -> list[str]:
        return [o.name for o in self.objects]

    def select(self, index: int) -> bool:
        index %= len(self.objects)
        changed = index != self.index
        self.index = index
        self._last_switch = time.monotonic()
        return changed

    def step(self, delta: int) -> bool:
        return self.select(self.index + delta)

    def toggle_shuffle(self) -> bool:
        self.shuffle_on = not self.shuffle_on
        if self.shuffle_on and self.shuffle_minutes <= 0:
            self.shuffle_minutes = 3.0
        self._last_switch = time.monotonic()
        return self.shuffle_on

    def maybe_shuffle(self) -> bool:
        """Switch to a random other object when the shuffle interval has passed."""
        if not self.shuffle_on or len(self.objects) < 2:
            return False
        if time.monotonic() - self._last_switch < self.shuffle_minutes * 60:
            return False
        choices = [i for i in range(len(self.objects)) if i != self.index]
        return self.select(random.choice(choices))
