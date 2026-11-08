"""Build curated templates for catalog objects from Quick, Draw! sketches.

For each object: stream the first N drawings of its simplified ndjson file
(cached in data/quickdraw/), keep recognized simple outlines, pick a set of
typical-but-diverse sketches, and save them to templates/<name>.npz together
with ranking statistics in templates/<name>.json.

    uv run python tools/build_templates.py               # catalog + reserves
    uv run python tools/build_templates.py fish key      # just these
    uv run python tools/build_templates.py fish --drop 3 17   # reject sketches 3 and 17, refill

Quick, Draw! data: https://github.com/googlecreativelab/quickdraw-dataset (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pareidolia.catalog import load_catalog  # noqa: E402
from pareidolia.geometry import (normalize, pack_sketches, sample_points,  # noqa: E402
                                 sketch_bbox, sketch_from_quickdraw, sketch_length,
                                 stroke_length)
from pareidolia.params import ROOT, load_config  # noqa: E402
from pareidolia.templates import ObjectTemplates  # noqa: E402

DATA_DIR = ROOT / "data" / "quickdraw"
TEMPLATE_DIR = ROOT / "templates"
URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"

RASTER = 64          # canvas size for sketch-to-sketch chamfer distances
MAX_POOL = 600       # candidates considered for selection


def download(name: str, n_lines: int) -> Path:
    """Stream the first n_lines drawings of a category into the cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.ndjson"
    if path.exists():
        with open(path, "rb") as f:
            have = sum(1 for _ in f)
        if have >= n_lines:
            return path
    url = URL.format(urllib.parse.quote(name))
    print(f"  downloading {n_lines} drawings of '{name}'", flush=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as out:
        for i, line in enumerate(r):
            if i >= n_lines:
                break
            out.write(line)
    tmp.replace(path)
    return path


def load_candidates(path: Path, args, excluded: set[str]):
    """Parse, filter and normalize drawings. Returns sketches, key_ids, stats."""
    n_read = n_recog = 0
    stroke_counts = []
    sketches, keys = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            n_read += 1
            if not d.get("recognized", False):
                continue
            n_recog += 1
            if str(d["key_id"]) in excluded:
                continue
            sk = [s for s in sketch_from_quickdraw(d["drawing"]) if len(s) >= 2]
            if not sk:
                continue
            total = sketch_length(sk)
            if total <= 0:
                continue
            # Drop tiny strokes (eyes, dots, short hatching): they rarely show up as real edges.
            sk = [s for s in sk if stroke_length(s) >= args.tiny * total]
            if not (1 <= len(sk) <= args.max_strokes):
                continue
            x0, y0, x1, y1 = sketch_bbox(sk)
            if min(x1 - x0, y1 - y0) < 0.15 * max(x1 - x0, y1 - y0):
                continue  # nearly a straight line
            sk = normalize(sk)
            ink = sketch_length(sk)
            if not (args.min_ink <= ink <= args.max_ink):
                continue  # too sparse or a scribble
            sketches.append(sk)
            keys.append(str(d["key_id"]))
            stroke_counts.append(len(sk))
    stats = {
        "drawings_read": n_read,
        "recognized_rate": n_recog / max(n_read, 1),
        "pass_rate": len(sketches) / max(n_recog, 1),
        "mean_strokes": float(np.mean(stroke_counts)) if stroke_counts else 0.0,
    }
    return sketches, keys, stats


def raster_points(sk) -> np.ndarray:
    """Normalized sketch -> integer points on the RASTER canvas."""
    scale = RASTER - 8
    pts, _ = sample_points([(s * scale) + RASTER / 2 for s in sk], 1.5, 8)
    return np.clip(np.round(pts).astype(np.int32), 0, RASTER - 1)


def distance_matrix(sketches) -> np.ndarray:
    """Symmetric mean chamfer distance between normalized sketches (in RASTER px)."""
    n = len(sketches)
    dts = np.empty((n, RASTER, RASTER), np.float32)
    pts = []
    for i, sk in enumerate(sketches):
        p = raster_points(sk)
        pts.append(p)
        canvas = np.full((RASTER, RASTER), 255, np.uint8)
        canvas[p[:, 1], p[:, 0]] = 0
        dts[i] = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
    m = np.empty((n, n), np.float32)
    for i, p in enumerate(pts):
        m[i] = dts[:, p[:, 1], p[:, 0]].mean(axis=1)
    return (m + m.T) / 2


def select(sketches, dist: np.ndarray, count: int, k: int = 10) -> list[int]:
    """Greedy pick: most typical first, each new pick far enough from the others."""
    n = len(sketches)
    k = min(k, n - 1)
    d = dist + np.eye(n) * 1e9
    knn = np.sort(d, axis=1)[:, :k]
    typicality = knn.mean(axis=1)            # low = many close neighbours
    order = np.argsort(typicality)
    spacing = float(np.median(knn[:, 0]))    # typical nearest-neighbour distance
    picked: list[int] = []
    min_sep = spacing
    while len(picked) < count and min_sep > 1e-3:
        for i in order:
            if len(picked) >= count:
                break
            if i in picked:
                continue
            # Only the more typical half of the pool is eligible.
            if typicality[i] > np.quantile(typicality, 0.5):
                continue
            if all(dist[i, j] >= min_sep for j in picked):
                picked.append(int(i))
        min_sep *= 0.8
    return picked


def shape_stats(sketches) -> dict:
    """Curvature measures on the picked sketches.

    bins_used: orientation bins holding at least 6% of the outline.
    straight_share: share of ink in long straight runs (polyline segments longer
    than 15% of the sketch size; the simplified data keeps curves as many short
    segments, so long segments are real straight lines).
    """
    bins_used, straight = [], []
    for sk in sketches:
        _, b = sample_points(sk, 0.01, 8)
        counts = np.bincount(b, minlength=8) / max(len(b), 1)
        bins_used.append(int((counts >= 0.06).sum()))
        seg = np.concatenate([np.hypot(*np.diff(s, axis=0).T) for s in sk if len(s) > 1])
        straight.append(float(seg[seg > 0.15].sum() / max(seg.sum(), 1e-9)))
    return {
        "bins_used": float(np.mean(bins_used)),
        "straight_share": float(np.mean(straight)) if straight else 1.0,
    }


def build(name: str, args) -> dict | None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    exclude_path = TEMPLATE_DIR / f"{name}.exclude.txt"
    excluded = set(exclude_path.read_text().split()) if exclude_path.exists() else set()

    path = download(name, args.lines)
    sketches, keys, stats = load_candidates(path, args, excluded)
    if len(sketches) < args.count:
        print(f"  ! only {len(sketches)} usable sketches for '{name}'")
        if len(sketches) < 5:
            return None
    rng = np.random.default_rng(0)
    pool = rng.permutation(len(sketches))[:MAX_POOL]
    sketches = [sketches[i] for i in pool]
    keys = [keys[i] for i in pool]

    dist = distance_matrix(sketches)
    sub = rng.permutation(len(sketches))[:200]
    iu = np.triu_indices(len(sub), 1)
    stats["consistency"] = float(np.median(dist[np.ix_(sub, sub)][iu]))

    picked = select(sketches, dist, args.count)
    chosen = [sketches[i] for i in picked]
    stats.update(shape_stats(chosen))
    stats["templates"] = len(chosen)
    stats["excluded"] = len(excluded)

    np.savez_compressed(TEMPLATE_DIR / f"{name}.npz", **pack_sketches(chosen),
                        key_ids=np.array([keys[i] for i in picked]))
    (TEMPLATE_DIR / f"{name}.json").write_text(json.dumps(stats, indent=2))

    # Prebuild the variant cache for catalog objects so the app starts fast.
    specs = {o.name: o for o in load_catalog().objects}
    if name in specs:
        ObjectTemplates(specs[name], chosen, load_config())
    return stats


def drop(name: str, indices: list[int]) -> None:
    """Add the key_ids of the given template indices to the exclude list."""
    data = np.load(TEMPLATE_DIR / f"{name}.npz")
    keys = [str(k) for k in data["key_ids"]]
    exclude_path = TEMPLATE_DIR / f"{name}.exclude.txt"
    existing = exclude_path.read_text().split() if exclude_path.exists() else []
    new = [keys[i] for i in indices if 0 <= i < len(keys)]
    exclude_path.write_text("\n".join(existing + new) + "\n")
    print(f"  excluded {len(new)} sketches from '{name}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="categories (default: catalog objects and reserves)")
    ap.add_argument("--catalog-only", action="store_true", help="skip the reserves")
    ap.add_argument("--lines", type=int, default=8000, help="drawings to read per category")
    ap.add_argument("--count", type=int, default=40, help="templates to keep per object")
    ap.add_argument("--max-strokes", type=int, default=4)
    ap.add_argument("--tiny", type=float, default=0.06, help="drop strokes shorter than this share of the ink")
    ap.add_argument("--min-ink", type=float, default=2.0)
    ap.add_argument("--max-ink", type=float, default=7.0)
    ap.add_argument("--drop", type=int, nargs="+", help="template indices to reject (one name only)")
    args = ap.parse_args()

    names = args.names
    if not names:
        cat = load_catalog()
        names = [o.name for o in cat.objects] + ([] if args.catalog_only else cat.reserves)
    if args.drop:
        if len(names) != 1:
            ap.error("--drop needs exactly one name")
        drop(names[0], args.drop)

    for name in names:
        print(f"{name}:", flush=True)
        try:
            s = build(name, args)
        except Exception as e:  # keep going with the other objects
            print(f"  ! failed: {e}")
            continue
        if s:
            print(f"  {s['templates']} templates  pass {s['pass_rate']:.0%}  "
                  f"consistency {s['consistency']:.2f}  bins {s['bins_used']:.1f}  "
                  f"straight {s['straight_share']:.0%}")


if __name__ == "__main__":
    main()
