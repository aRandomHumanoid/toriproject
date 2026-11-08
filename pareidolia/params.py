"""Configuration loading and the suggestibility mapping."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "camera": {
        "device": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "mirror": False,
        "lock_exposure": False,
    },
    "processing": {
        "width": 640,
        "height": 360,
        "orientation_bins": 8,
        "tau": 8,
        "tau_coarse": 4,
        "delta": 2,
        "blur_sigma": 1.2,
        "min_fragment": 8,
    },
    "search": {
        "budget_ms": 14.0,
        "coarse_stride": 2,
        "coarse_slack": 0.10,
        "max_candidates": 20,
        "refine_radius": 4,
        "scale_min": 0.18,
        "scale_max": 0.60,
        "scale_steps": 5,
        "rotation_step": 15,
        "mirror": True,
        "point_spacing": 3.0,
        "early_fraction": 0.15,
        "size_bias": 0.12,
    },
    "match": {
        "suggestibility": 0.5,
        "threshold_low": 0.20,
        "threshold_high": 0.50,
        "coverage_low": 0.35,
        "coverage_high": 0.75,
        "clutter_low": 0.06,
        "clutter_high": 0.16,
        "clutter_penalty": 2.0,
        "clutter_rho0": 0.03,
        "bin_span": 0.5,
        "cross_penalty": 1.0,
        "cross_free": 0.08,
    },
    "track": {
        "max_tracks": 3,
        "max_tentative": 6,
        "confirm_frames": 3,
        "max_misses": 10,
        "fade_in_s": 0.3,
        "fade_out_s": 0.6,
        "max_lifetime_s": 8.0,
        "cooldown_s": 3.0,
        "local_radius": 3,
        "hysteresis": 0.05,
        "overlap_iou": 0.3,
        "smooth_min_cutoff": 1.0,
        "smooth_beta": 0.02,
    },
    "render": {
        "line_color": [255, 255, 255],
        "line_width": 2,
        "glow": 6.0,
        "glow_strength": 0.5,
        "ghost_opacity": 0.3,
        "ghost_width": 1,
        "background_dim": 0.3,
        "background_desaturate": 0.0,
        "fullscreen": False,
        "hud_seconds": 3.0,
    },
    "catalog": {
        "object": "fish",
        "shuffle_minutes": 0.0,
        "sighting_target": 3.0,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """Nested dict with attribute-style section access: cfg.search['budget_ms']."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.toml"
    data = {}
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
    return Config(_merge(DEFAULTS, data))


@dataclass(frozen=True)
class Limits:
    """Acceptance limits for one frame."""

    threshold: float      # theta: highest adjusted cost accepted
    coverage: float       # c_min: share of outline points that must be on an edge
    clutter_max: float    # rho_max: edge density above which a region is ignored
    clutter_penalty: float
    clutter_rho0: float
    bin_span: float

    def relaxed(self, by: float) -> "Limits":
        """Looser limits for keeping an existing track alive (hysteresis)."""
        return Limits(self.threshold + by, max(0.0, self.coverage - 2 * by),
                      self.clutter_max + by, self.clutter_penalty, self.clutter_rho0,
                      max(0.0, self.bin_span - by))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def limits_for(cfg: Config, suggestibility: float, offset: float = 0.0) -> Limits:
    """Map suggestibility s in [0, 1] to matching limits; offset is the object's
    calibrated threshold offset."""
    s = min(max(suggestibility, 0.0), 1.0)
    m = cfg.match
    return Limits(
        threshold=lerp(m["threshold_low"], m["threshold_high"], s) + offset,
        coverage=lerp(m["coverage_high"], m["coverage_low"], s),
        clutter_max=lerp(m["clutter_low"], m["clutter_high"], s),
        clutter_penalty=m["clutter_penalty"],
        clutter_rho0=m["clutter_rho0"],
        bin_span=m["bin_span"],
    )
