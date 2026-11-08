"""Template variants: every curated sketch at every scale, rotation and mirror,
as integer point lists at the fine (640x360) and coarse (320x180) levels.

Point lists are packed CSR-style so the Numba kernels can index them:
    pts[start[v] : start[v] + count[v]] = (dx, dy, bin) relative to the variant's
    top-left corner; (cx[v], cy[v]) is the sketch center inside that box.
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .catalog import ObjectSpec
from .geometry import Sketch, sample_points, unpack_sketches
from .params import ROOT, Config

TEMPLATE_DIR = ROOT / "templates"
CACHE_DIR = TEMPLATE_DIR / "cache"
CACHE_VERSION = 2


@dataclass
class Level:
    """Packed variant points at one resolution."""

    start: np.ndarray    # (V,) int64
    count: np.ndarray    # (V,) int32
    w: np.ndarray        # (V,) int32 box width
    h: np.ndarray        # (V,) int32 box height
    cx: np.ndarray       # (V,) int32 sketch center inside the box
    cy: np.ndarray       # (V,) int32
    binmask: np.ndarray  # (V,) int32 bitmask of orientation bins used
    pts: np.ndarray      # (N, 3) int16 (dx, dy, bin)


def _pack(points: list[np.ndarray]) -> Level:
    V = len(points)
    start = np.zeros(V, np.int64)
    count = np.zeros(V, np.int32)
    w = np.zeros(V, np.int32)
    h = np.zeros(V, np.int32)
    cx = np.zeros(V, np.int32)
    cy = np.zeros(V, np.int32)
    binmask = np.zeros(V, np.int32)
    chunks = []
    pos = 0
    for v, (p, c) in enumerate(points):
        start[v] = pos
        count[v] = len(p)
        w[v] = p[:, 0].max() + 1
        h[v] = p[:, 1].max() + 1
        cx[v], cy[v] = c
        binmask[v] = int(np.bitwise_or.reduce(1 << p[:, 2].astype(np.int64)))
        chunks.append(p)
        pos += len(p)
    pts = np.concatenate(chunks, axis=0).astype(np.int16)
    return Level(start, count, w, h, cx, cy, binmask, pts)


def _variant_points(sketch: Sketch, scale: float, angle: float, mirror: bool,
                    spacing: float, n_bins: int, rng: np.random.Generator):
    """Integer points of one posed sketch, shifted to a top-left origin, shuffled
    (so a partial sum is an unbiased early estimate)."""
    a = np.radians(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    posed = []
    for s in sketch:
        p = s.copy()
        if mirror:
            p[:, 0] = -p[:, 0]
        posed.append((p * scale) @ rot.T)
    P, B = sample_points(posed, spacing, n_bins)
    xy = np.round(P).astype(np.int64)
    trip = np.unique(np.column_stack([xy, B]), axis=0)
    x0, y0 = trip[:, 0].min(), trip[:, 1].min()
    trip[:, 0] -= x0
    trip[:, 1] -= y0
    trip = trip[rng.permutation(len(trip))]
    return trip, (int(-x0), int(-y0))


class ObjectTemplates:
    """All variants of one object."""

    def __init__(self, spec: ObjectSpec, sketches: list[Sketch], cfg: Config):
        self.spec = spec
        self.name = spec.name
        self.sketches = sketches
        sc, pc = cfg.search, cfg.processing
        H = pc["height"]
        self.n_bins = pc["orientation_bins"]
        self.scales = np.geomspace(sc["scale_min"] * H, sc["scale_max"] * H, sc["scale_steps"])
        step = sc["rotation_step"]
        if spec.full_rotation:
            self.rotations = np.arange(0, 360, step, dtype=float)
        else:
            r = spec.rotation
            self.rotations = np.arange(-r, r + 1e-6, step, dtype=float)
            if len(self.rotations) == 0:
                self.rotations = np.array([0.0])
        self.wrap_rotation = spec.full_rotation
        self.mirrors = [False, True] if sc["mirror"] else [False]
        self.S, self.M = len(sketches), len(self.mirrors)
        self.NS, self.NR = len(self.scales), len(self.rotations)
        self.n_variants = self.S * self.M * self.NS * self.NR
        self.spacing = sc["point_spacing"]
        self._build(cfg)

    # Variant index <-> (sketch, mirror, scale, rotation)
    def index(self, s: int, m: int, si: int, ri: int) -> int:
        return ((s * self.M + m) * self.NS + si) * self.NR + ri

    def unindex(self, v: int) -> tuple[int, int, int, int]:
        v, ri = divmod(int(v), self.NR)
        v, si = divmod(v, self.NS)
        s, m = divmod(v, self.M)
        return s, m, si, ri

    def pose(self, v: int) -> tuple[int, bool, float, float]:
        """(sketch, mirrored, scale in fine px, angle in degrees)."""
        s, m, si, ri = self.unindex(v)
        return s, self.mirrors[m], float(self.scales[si]), float(self.rotations[ri])

    def neighbors(self, v: int) -> list[int]:
        """Same sketch and mirror, scale and rotation within one step."""
        s, m, si, ri = self.unindex(v)
        out = []
        for dsi in (-1, 0, 1):
            nsi = si + dsi
            if not 0 <= nsi < self.NS:
                continue
            for dri in (-1, 0, 1):
                nri = ri + dri
                if self.wrap_rotation:
                    nri %= self.NR
                elif not 0 <= nri < self.NR:
                    continue
                out.append(self.index(s, m, nsi, nri))
        return sorted(set(out))

    def _cache_key(self) -> str:
        h = hashlib.sha1()
        h.update(f"{CACHE_VERSION}|{self.scales.tolist()}|{self.rotations.tolist()}|"
                 f"{self.mirrors}|{self.spacing}|{self.n_bins}".encode())
        for sk in self.sketches:
            for s in sk:
                h.update(np.ascontiguousarray(s, np.float64).tobytes())
        return h.hexdigest()[:16]

    def _build(self, cfg: Config) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"{self.name}-{self._cache_key()}.npz"
        if cache.exists():
            try:
                d = np.load(cache)
                self.fine = Level(*(d[f"f_{k}"] for k in Level.__dataclass_fields__))
                self.coarse = Level(*(d[f"c_{k}"] for k in Level.__dataclass_fields__))
                return
            except Exception:
                pass
        rng = np.random.default_rng(1)
        fine, coarse = [], []
        for v in range(self.n_variants):
            s, mirror, scale, angle = self.pose(v)
            sk = self.sketches[s]
            fine.append(_variant_points(sk, scale, angle, mirror, self.spacing, self.n_bins, rng))
            coarse.append(_variant_points(sk, scale / 2, angle, mirror, self.spacing / 1.5,
                                          self.n_bins, rng))
        self.fine = _pack(fine)
        self.coarse = _pack(coarse)
        arrays = {f"f_{k}": getattr(self.fine, k) for k in Level.__dataclass_fields__}
        arrays.update({f"c_{k}": getattr(self.coarse, k) for k in Level.__dataclass_fields__})
        for old in CACHE_DIR.glob(f"{self.name}-*.npz"):
            old.unlink(missing_ok=True)
        np.savez(cache, **arrays)


def load_sketches(name: str) -> list[Sketch]:
    path = TEMPLATE_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no templates for '{name}'. Run: uv run python tools/build_templates.py \"{name}\"")
    return unpack_sketches(np.load(path))


class TemplateStore:
    """Loads every catalog object's variants: the requested one first, the rest
    in a background thread so switching objects is instant."""

    def __init__(self, specs: list[ObjectSpec], cfg: Config, first: str | None = None):
        self.cfg = cfg
        self.specs = {s.name: s for s in specs}
        self._lock = threading.Lock()
        self._ready: dict[str, ObjectTemplates] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="templates")
        self._futures: dict[str, Future] = {}
        self.missing: list[str] = []
        self.sketches: dict[str, list[Sketch]] = {}
        for s in specs:
            try:
                self.sketches[s.name] = load_sketches(s.name)
            except FileNotFoundError:
                self.missing.append(s.name)
        order = [first] if first in self.sketches else []
        order += [n for n in self.sketches if n not in order]
        for i, n in enumerate(order):
            if i == 0:
                self._ready[n] = ObjectTemplates(self.specs[n], self.sketches[n], cfg)
            else:
                self._futures[n] = self._pool.submit(self._build, n)

    def _build(self, name: str) -> ObjectTemplates:
        t = ObjectTemplates(self.specs[name], self.sketches[name], self.cfg)
        with self._lock:
            self._ready[name] = t
        return t

    def available(self, name: str) -> bool:
        return name in self.sketches

    def get(self, name: str) -> ObjectTemplates:
        with self._lock:
            if name in self._ready:
                return self._ready[name]
        if name in self._futures:
            return self._futures[name].result()
        return self._build(name)

    def preview(self, name: str) -> Sketch | None:
        """The most typical sketch of an object, for the picker."""
        sk = self.sketches.get(name)
        return sk[0] if sk else None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
