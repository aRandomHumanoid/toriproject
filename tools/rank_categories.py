"""Rank candidate objects and optionally rewrite the catalog with the best 20.

    uv run python tools/rank_categories.py            # print the ranking
    uv run python tools/rank_categories.py --write    # keep the top 20 in catalog.toml

Measures (from templates/<name>.json, written by build_templates.py, and
templates/calibration.json, written by calibrate.py --reserves):
  consistency    median distance between random sketches (low = people agree)
  simplicity     share of recognized drawings that are simple outlines (high)
  curvature      orientation bins used (high) and share of long straight runs (low)
  sighting rate  raw sightings/min on the reference clips; closest to the target wins
Each measure is ranked; the combined score is the mean rank.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pareidolia.catalog import ObjectSpec, load_catalog, save_catalog  # noqa: E402
from pareidolia.params import ROOT, load_config  # noqa: E402

# Defaults for candidates that are not in the catalog yet.
DEFAULTS = {
    "ear": ("Body", 20), "shark": ("Animal", 30), "duck": ("Animal", 30), "flamingo": ("Animal", 15),
    "octopus": ("Animal", 30), "palm tree": ("Plant", 15), "pear": ("Plant", 30), "crown": ("Object", 15),
    "sailboat": ("Object", 15), "scissors": ("Object", 90), "skull": ("Body", 15),
    "hot air balloon": ("Object", 10),
}


def ranks(values: dict[str, float], higher_better: bool) -> dict[str, int]:
    order = sorted(values, key=lambda k: values[k], reverse=higher_better)
    return {k: i + 1 for i, k in enumerate(order)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="rewrite catalog.toml with the top N")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    cat = load_catalog()
    names = [o.name for o in cat.objects] + cat.reserves
    stats = {}
    for n in names:
        p = ROOT / "templates" / f"{n}.json"
        if p.exists():
            stats[n] = json.loads(p.read_text())
        else:
            print(f"(no stats for '{n}': run build_templates.py)")
    cal_path = ROOT / "templates" / "calibration.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    target = float(load_config().catalog["sighting_target"])

    names = list(stats)
    r_cons = ranks({n: stats[n]["consistency"] for n in names}, higher_better=False)
    r_simp = ranks({n: stats[n]["pass_rate"] for n in names}, higher_better=True)
    r_bins = ranks({n: stats[n]["bins_used"] for n in names}, higher_better=True)
    r_str = ranks({n: stats[n]["straight_share"] for n in names}, higher_better=False)
    have_rate = [n for n in names if n in cal]
    r_rate = ranks({n: abs(math.log((cal[n]["rate_at_zero"] + 0.1) / target)) for n in have_rate},
                   higher_better=False) if have_rate else {}

    score = {}
    for n in names:
        parts = [r_cons[n], r_simp[n], (r_bins[n] + r_str[n]) / 2]
        if n in r_rate:
            parts.append(r_rate[n] * len(names) / max(len(have_rate), 1))
        score[n] = sum(parts) / len(parts)
    order = sorted(names, key=lambda n: score[n])

    in_cat = {o.name for o in cat.objects}
    print(f"{'#':>2}  {'object':16s} {'score':>5}  {'consist':>7} {'simple':>6} {'bins':>4} "
          f"{'straight':>8} {'rate/min':>8}  catalog")
    for i, n in enumerate(order, 1):
        s = stats[n]
        rate = f"{cal[n]['rate_at_zero']:.1f}" if n in cal else "-"
        mark = "yes" if n in in_cat else ""
        cut = "  ---- cut" if i == args.top else ""
        print(f"{i:2d}  {n:16s} {score[n]:5.1f}  {s['consistency']:7.2f} {s['pass_rate']:6.0%} "
              f"{s['bins_used']:4.1f} {s['straight_share']:8.0%} {rate:>8}  {mark}{cut}")
    if not have_rate:
        print("\nno sighting rates yet: run tools/calibrate.py --reserves for the full ranking")

    if args.write:
        specs = {o.name: o for o in cat.objects}
        top = order[:args.top]
        new = []
        for n in top:
            if n in specs:
                new.append(specs[n])
            else:
                group, rot = DEFAULTS.get(n, ("Object", 30))
                new.append(ObjectSpec(n, group, rot, 0.0))
        cat.objects = sorted(new, key=lambda o: (o.group, o.name))
        cat.reserves = [n for n in order[args.top:]]
        save_catalog(cat)
        print(f"\ncatalog.toml now holds the top {len(top)}; run build_templates.py for any new ones")


if __name__ == "__main__":
    main()
