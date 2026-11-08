"""Per-stage timing for each catalog object on a clip, a still image or the camera.

    uv run python tools/bench.py                                   # a built-in test photo
    uv run python tools/bench.py --source clips/clip_x.mp4 --frames 300
    uv run python tools/bench.py --source 0 --objects fish key     # live camera
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pareidolia  # noqa: E402,F401
import cv2  # noqa: E402

from pareidolia.app import Pipeline  # noqa: E402
from pareidolia.capture import open_source  # noqa: E402
from pareidolia.catalog import Catalog, load_catalog  # noqa: E402
from pareidolia.matcher import warmup  # noqa: E402
from pareidolia.params import load_config  # noqa: E402
from pareidolia.templates import TemplateStore  # noqa: E402

DEFAULT_SOURCE = "/usr/share/backgrounds/osselo-Ask_a_friend.jpg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--objects", nargs="*")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    cv2.setNumThreads(4)
    cfg = load_config()
    specs = load_catalog().objects
    if args.objects:
        specs = [s for s in specs if s.name in args.objects]
    store = TemplateStore(specs, cfg, first=specs[0].name)
    for s in specs:            # finish loading before timing anything
        store.get(s.name)
    warmup()
    print(f"{'object':14s} {'variants':>8} {'edges':>6} {'track':>6} {'sweep':>6} {'refine':>6} "
          f"{'render':>6} {'frame':>6} {'fps':>5} {'full sweep':>10}")
    for spec in specs:
        catalog = Catalog([spec])
        src = open_source(args.source, cfg, realtime=False)
        pipe = Pipeline(cfg, store, catalog)
        for _ in range(30):                     # settle the adaptive budget
            pipe.step(src.read())
        pipe.timing.clear()
        for _ in range(args.frames):
            pipe.step(src.read())
        src.close()
        t = pipe.timing
        period = pipe.matcher.sweep_period_s
        print(f"{spec.name:14s} {pipe.templates.n_variants:8d} {t['edges']:6.1f} {t['track']:6.1f} "
              f"{t['sweep']:6.1f} {t['refine']:6.1f} {t['render']:6.1f} {t['frame']:6.1f} "
              f"{1000 / t['frame']:5.0f} {period if period else float('nan'):9.2f}s")
    store.close()


if __name__ == "__main__":
    main()
