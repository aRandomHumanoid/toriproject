"""Render curated templates as contact sheets for review.

    uv run python tools/contact_sheet.py fish          # every template of one object
    uv run python tools/contact_sheet.py --catalog     # the most typical sketch of each catalog object

Sheets are written to data/contact_sheets/. Reject bad sketches with
    uv run python tools/build_templates.py fish --drop 3 17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pareidolia.catalog import load_catalog  # noqa: E402
from pareidolia.geometry import unpack_sketches  # noqa: E402
from pareidolia.params import ROOT  # noqa: E402

OUT = ROOT / "data" / "contact_sheets"


def draw_tile(sketch, label: str, size: int = 160) -> np.ndarray:
    tile = np.full((size, size, 3), 24, np.uint8)
    scale = size * 0.72
    for s in sketch:
        p = np.round(s * scale + [size / 2, size / 2 - 8]).astype(np.int32)
        cv2.polylines(tile, [p], False, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(tile, label, (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 255), 1, cv2.LINE_AA)
    return tile


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return sheet


def load(name: str):
    return unpack_sketches(np.load(ROOT / "templates" / f"{name}.npz"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--catalog", action="store_true", help="one sheet with the first sketch of every object")
    ap.add_argument("--reserves", action="store_true", help="with --catalog: include reserves")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.catalog:
        cat = load_catalog()
        names = [o.name for o in cat.objects] + (cat.reserves if args.reserves else [])
        tiles = []
        for n in names:
            if (ROOT / "templates" / f"{n}.npz").exists():
                tiles.append(draw_tile(load(n)[0], n))
        path = OUT / "catalog.png"
        cv2.imwrite(str(path), grid(tiles, 8))
        print(path)
    for name in args.names:
        sk = load(name)
        tiles = [draw_tile(s, f"{i}") for i, s in enumerate(sk)]
        path = OUT / f"{name}.png"
        cv2.imwrite(str(path), grid(tiles, 10))
        print(path)


if __name__ == "__main__":
    main()
