"""Calibrate per-object threshold offsets so every object appears at about the
same rate: catalog.sighting_target sightings per minute at suggestibility 0.5.

    uv run python tools/calibrate.py                         # every clip in clips/
    uv run python tools/calibrate.py --objects fish key      # just these
    uv run python tools/calibrate.py --seconds 20 --step 3   # quicker, rougher

Runs the real pipeline (no rendering) over the reference clips in simulated
time, bisecting each object's offset. Writes the offsets to catalog.toml and the
raw rates to templates/calibration.json (used by rank_categories.py).
Record clips first with tools/record_clip.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pareidolia  # noqa: E402,F401
import cv2  # noqa: E402

from pareidolia.app import Pipeline  # noqa: E402
from pareidolia.capture import IMAGE_EXT, VIDEO_EXT, StillImage, VideoFile  # noqa: E402
from pareidolia.catalog import Catalog, ObjectSpec, load_catalog, save_catalog  # noqa: E402
from pareidolia.matcher import warmup  # noqa: E402
from pareidolia.params import ROOT, load_config  # noqa: E402
from pareidolia.templates import ObjectTemplates, load_sketches  # noqa: E402

CAL_PATH = ROOT / "templates" / "calibration.json"


class OneStore:
    """Minimal template store for one object (no background threads)."""

    def __init__(self, t: ObjectTemplates):
        self.t = t

    def get(self, name: str) -> ObjectTemplates:
        return self.t


def sightings_per_minute(cfg, templates, spec, offset, sources, seconds, step) -> float:
    spec = dataclasses.replace(spec, threshold_offset=offset)
    pipe = Pipeline(cfg, OneStore(templates), Catalog([spec]), adaptive=False)
    pipe.suggestibility = 0.5
    total_s = 0.0
    sightings = 0
    for src_factory, fps in sources:
        src = src_factory()
        pipe.tracker.reset()
        pipe.last_now = None
        pipe.prev = None
        before = pipe.tracker.sightings
        n_frames = int(seconds * fps)
        t = 0.0
        for i in range(n_frames):
            frame = src.read()
            if frame is None:
                break
            if i % step:
                continue
            pipe.step(frame, now=t, render=False)
            t = (i + step) / fps
        src.close()
        total_s += t
        sightings += pipe.tracker.sightings - before
    return sightings / max(total_s / 60.0, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="*", help="default: every catalog object")
    ap.add_argument("--reserves", action="store_true", help="also measure reserves (rates only)")
    ap.add_argument("--clips", nargs="*", help="video files or images (default: everything in clips/)")
    ap.add_argument("--seconds", type=float, default=60, help="seconds used from each clip")
    ap.add_argument("--step", type=int, default=2, help="process every n-th frame")
    ap.add_argument("--target", type=float, help="sightings per minute (default: config)")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="do not write catalog.toml")
    args = ap.parse_args()

    cv2.setNumThreads(4)
    cfg = load_config()
    target = args.target or float(cfg.catalog["sighting_target"])
    size = (cfg.camera["width"], cfg.camera["height"])
    paths = [Path(p) for p in args.clips] if args.clips else sorted(
        p for p in (ROOT / "clips").glob("*") if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT)
    if not paths:
        sys.exit("no clips. Record some with: uv run python tools/record_clip.py")
    sources = []
    for p in paths:
        if p.suffix.lower() in IMAGE_EXT:
            sources.append((lambda p=p: StillImage(p, size, realtime=False), 30.0))
        else:
            fps = VideoFile(p, size, realtime=False).fps
            sources.append((lambda p=p: VideoFile(p, size, realtime=False), fps))

    cat = load_catalog()
    specs = {o.name: o for o in cat.objects}
    names = args.objects or list(specs)
    if args.reserves:
        names += [r for r in cat.reserves if r not in names]
    warmup()
    results = json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else {}
    print(f"target {target:.1f} sightings/min over {len(paths)} clip(s), {args.seconds:.0f} s each")

    for name in names:
        spec = specs.get(name, ObjectSpec(name))
        try:
            templates = ObjectTemplates(spec, load_sketches(name), cfg)
        except FileNotFoundError as e:
            print(f"{name}: {e}")
            continue
        t0 = time.time()
        rate0 = sightings_per_minute(cfg, templates, spec, 0.0, sources, args.seconds, args.step)
        entry = {"rate_at_zero": rate0}
        if name in specs:
            lo, hi = (-0.15, 0.0) if rate0 > target else (0.0, 0.20)
            best, best_err = 0.0, abs(rate0 - target)
            for _ in range(args.iterations):
                mid = (lo + hi) / 2
                r = sightings_per_minute(cfg, templates, spec, mid, sources, args.seconds, args.step)
                if abs(r - target) < best_err:
                    best, best_err = mid, abs(r - target)
                if r > target:
                    hi = mid
                else:
                    lo = mid
            specs[name].threshold_offset = round(best, 3)
            entry["offset"] = round(best, 3)
        results[name] = entry
        print(f"{name:16s} raw {rate0:5.1f}/min  offset {entry.get('offset', 0):+.3f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        CAL_PATH.write_text(json.dumps(results, indent=2))

    if not args.dry_run:
        save_catalog(cat)
        print("offsets written to catalog.toml")


if __name__ == "__main__":
    main()
