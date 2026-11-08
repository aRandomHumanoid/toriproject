"""Oriented chamfer matching: a time-budgeted coarse sweep over a slice of the
variant library, then refinement of the best candidates at full resolution."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .edges import Features
from .params import Config, Limits
from .templates import ObjectTemplates

TOPK = 3  # candidates kept per variant per sweep


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _sweep_kernel(D, H, W, integ, var_ids, start, count, w, h, tlen, spen, off, pdx, pdy, pbin, nbins,
                  stride, tau, thr, early_frac, lam, rho0, rho_max, mu, xfree, out_cost, out_x, out_y):
    """Score every variant in var_ids over a position grid on the coarse maps.

    D is the flattened (K, H, W) distance stack; off[i] = bin*H*W + dy*W + dx.
    1. Vectorized early pass: the first n_early (shuffled) points are summed for
       all positions at once, as shifted row adds the CPU can run in SIMD.
    2. Per position: excess edge density first (4 lookups: edges in the box
       beyond the outline's own length tlen, per box area); the clutter penalty
       tightens the cost limit; then the early sum, then the remaining points
       with a running limit.
    3. Survivors pay a crossing penalty mu * max(0, cross - xfree), where cross is
       the share of points sitting on a near-perpendicular edge.
    Keeps the best TOPK well-separated positions per variant."""
    topk = out_cost.shape[1]
    for j in prange(var_ids.shape[0]):
        v = var_ids[j]
        s = start[v]
        n = count[v]
        vw = w[v]
        vh = h[v]
        for t in range(topk):
            out_cost[j, t] = 1e9
            out_x[j, t] = -1
            out_y[j, t] = -1
        if vw >= W or vh >= H or n == 0:
            continue
        n_early = max(6, int(n * early_frac))
        if n_early > n:
            n_early = n
        rad = 0.5 * max(vw, vh)
        area = 4.0 * vw * vh
        own = tlen[v]
        sp = spen[v]
        Hp = H - vh
        Wp = W - vw
        acc = np.zeros((Hp, Wp), np.int16)
        for i in range(s, s + n_early):
            b = pbin[i]
            dx = pdx[i]
            dy = pdy[i]
            for y in range(0, Hp, stride):
                row = (b * H + y + dy) * W + dx
                for x in range(Wp):
                    acc[y, x] += D[row + x]
        for y in range(0, Hp, stride):
            fy0 = 2 * y
            fy1 = fy0 + 2 * vh
            for x in range(0, Wp, stride):
                fx0 = 2 * x
                fx1 = fx0 + 2 * vw
                cnt = integ[fy1, fx1] - integ[fy0, fx1] - integ[fy1, fx0] + integ[fy0, fx0]
                rho = max(0.0, cnt - own) / area
                if rho > rho_max:
                    continue
                pen = lam * max(0.0, rho - rho0) + sp
                budget = thr - pen
                if budget <= 0.0:
                    continue
                a = acc[y, x]
                if a > budget * 1.2 * n_early * tau:
                    continue
                lim_full = budget * n * tau
                base = y * W + x
                tot = np.int64(a)
                i = s + n_early
                while i < s + n and tot <= lim_full:
                    tot += D[base + off[i]]
                    i += 1
                if tot > lim_full:
                    continue
                sc = tot / (n * tau) + pen
                if mu > 0.0:
                    crossed = 0
                    plane = H * W
                    for i in range(s, s + n):
                        pb = pbin[i]
                        perp = (pb + nbins // 2) % nbins
                        if D[base + off[i] + (perp - pb) * plane] == 0:
                            crossed += 1
                    sc += mu * max(0.0, crossed / n - xfree)
                    if sc > thr:
                        continue
                near = -1
                worst = 0
                worst_c = -1.0
                for t in range(topk):
                    c = out_cost[j, t]
                    if c > worst_c:
                        worst_c = c
                        worst = t
                    if out_x[j, t] >= 0 and abs(out_x[j, t] - x) < rad and abs(out_y[j, t] - y) < rad:
                        near = t
                if near >= 0:
                    if sc < out_cost[j, near]:
                        out_cost[j, near] = sc
                        out_x[j, near] = x
                        out_y[j, near] = y
                elif sc < worst_c:
                    out_cost[j, worst] = sc
                    out_x[j, worst] = x
                    out_y[j, worst] = y


@njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _refine_kernel(D, H, W, integ, cand_vars, cand_cx, cand_cy, radius, start, count, w, h, cx, cy,
                   binmask, tlen, spen, off, pbin, nbins, tau, delta, lam, rho0, mu, xfree, out):
    """For each candidate, search its neighbour variants within +/- radius px on
    the fine maps. out[c] = (variant, cx, cy, S, cost, coverage, rho, span, cross).
    cross = share of outline points with a near-perpendicular edge within 1 px:
    texture cuts across an outline, a clean outline is rarely crossed."""
    for c in prange(cand_vars.shape[0]):
        best = 1e9
        bv = -1
        bx = 0
        by = 0
        bcost = 0.0
        brho = 0.0
        for k in range(cand_vars.shape[1]):
            v = cand_vars[c, k]
            if v < 0:
                continue
            s = start[v]
            n = count[v]
            vw = w[v]
            vh = h[v]
            area = float(vw * vh)
            for dy in range(-radius, radius + 1):
                y0 = cand_cy[c] + dy - cy[v]
                if y0 < 0 or y0 + vh > H:
                    continue
                for dx in range(-radius, radius + 1):
                    x0 = cand_cx[c] + dx - cx[v]
                    if x0 < 0 or x0 + vw > W:
                        continue
                    base = y0 * W + x0
                    acc = 0
                    for i in range(s, s + n):
                        acc += D[base + off[i]]
                    cost = acc / (n * tau)
                    cnt = integ[y0 + vh, x0 + vw] - integ[y0, x0 + vw] - integ[y0 + vh, x0] + integ[y0, x0]
                    rho = max(0.0, cnt - tlen[v]) / area
                    sc = cost + lam * max(0.0, rho - rho0) + spen[v]
                    if sc < best:
                        best = sc
                        bv = v
                        bx = cand_cx[c] + dx
                        by = cand_cy[c] + dy
                        bcost = cost
                        brho = rho
        out[c, 0] = bv
        out[c, 1] = bx
        out[c, 2] = by
        out[c, 3] = best
        out[c, 4] = bcost
        out[c, 5] = 0.0
        out[c, 6] = brho
        out[c, 7] = 0.0
        out[c, 8] = 0.0
        if bv >= 0:
            s = start[bv]
            n = count[bv]
            base = (by - cy[bv]) * W + (bx - cx[bv])
            covered = 0
            crossed = 0
            cmask = 0
            plane = H * W
            for i in range(s, s + n):
                if D[base + off[i]] <= delta:
                    covered += 1
                    cmask |= 1 << pbin[i]
                pb = pbin[i]
                perp = (pb + nbins // 2) % nbins
                if D[base + off[i] + (perp - pb) * plane] <= 1:
                    crossed += 1
            out[c, 5] = covered / n
            out[c, 8] = crossed / n
            out[c, 3] += mu * max(0.0, crossed / n - xfree)
            used = _popcount(binmask[bv])
            out[c, 7] = _popcount(cmask) / max(used, 1)


def flat_offsets(pts: np.ndarray, H: int, W: int) -> np.ndarray:
    """(dx, dy, bin) -> offset into a flattened (K, H, W) array."""
    p = pts.astype(np.int64)
    return (p[:, 2] * H * W + p[:, 1] * W + p[:, 0]).astype(np.int64)


@dataclass
class Detection:
    variant: int
    cx: int          # sketch center, fine px
    cy: int
    score: float     # S = cost + clutter penalty
    cost: float
    coverage: float
    rho: float
    span: float
    box: tuple[int, int, int, int]  # x, y, w, h in fine px
    cross: float = 0.0  # share of outline points crossed by a perpendicular edge

    def accepted(self, lim: Limits) -> bool:
        return (self.variant >= 0 and self.score <= lim.threshold and self.coverage >= lim.coverage
                and self.rho <= lim.clutter_max and self.span >= lim.bin_span)


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Matcher:
    def __init__(self, cfg: Config):
        sc, pc = cfg.search, cfg.processing
        self.stride = int(sc["coarse_stride"])
        self.slack = float(sc["coarse_slack"])
        self.max_candidates = int(sc["max_candidates"])
        self.refine_radius = int(sc["refine_radius"])
        self.early = float(sc["early_fraction"])
        self.budget_ms = float(sc["budget_ms"])
        self.size_bias = float(sc["size_bias"])
        self.cross_penalty = float(cfg.match["cross_penalty"])
        self.cross_free = float(cfg.match["cross_free"])
        self.tau = int(pc["tau"])
        self.n_bins = int(pc["orientation_bins"])
        self.tau_c = int(pc["tau_coarse"])
        self.delta = int(pc["delta"])
        self.Hf, self.Wf = int(pc["height"]), int(pc["width"])
        self.Hc, self.Wc = self.Hf // 2, self.Wf // 2
        self._lam = float(cfg.match["clutter_penalty"])
        self._rho0 = float(cfg.match["clutter_rho0"])
        self.templates: ObjectTemplates | None = None
        self.rate = 20.0          # variants per ms, learned
        self.chunk = 64
        self.last_sweep_ms = 0.0
        self.last_refine_ms = 0.0
        self.sweep_period_s: float | None = None  # time to cover the whole library once
        self._order = np.zeros(0, np.int64)
        self._pos = 0
        self._cycle_start = time.perf_counter()
        self._rng = np.random.default_rng(2)

    def set_templates(self, t: ObjectTemplates) -> None:
        self.templates = t
        self._order = self._rng.permutation(t.n_variants).astype(np.int64)
        self._off_c = flat_offsets(t.coarse.pts, self.Hc, self.Wc)
        self._pdx_c = t.coarse.pts[:, 0].astype(np.int64)
        self._pdy_c = t.coarse.pts[:, 1].astype(np.int64)
        self._pbin_c = t.coarse.pts[:, 2].astype(np.int64)
        self._off_f = flat_offsets(t.fine.pts, self.Hf, self.Wf)
        self._pbin_f = t.fine.pts[:, 2].astype(np.int64)
        # Outline length in fine px: the edges a perfect match itself puts in its box.
        self._tlen = t.fine.count.astype(np.float64) * t.spacing
        # Size penalty: small outlines fit clutter by chance far more often.
        si = np.array([t.unindex(v)[2] for v in range(t.n_variants)])
        frac = si / max(t.NS - 1, 1)
        self._spen = (self.size_bias * (1.0 - frac)).astype(np.float64)
        self._pos = 0
        self._cycle_start = time.perf_counter()
        self.sweep_period_s = None

    # ------------------------------------------------------------------ sweep
    def _next_ids(self, n: int) -> np.ndarray:
        N = len(self._order)
        n = min(n, N)
        idx = (self._pos + np.arange(n)) % N
        self._pos += n
        if self._pos >= N:
            now = time.perf_counter()
            self.sweep_period_s = now - self._cycle_start
            self._cycle_start = now
            self._pos %= N
        return self._order[idx]

    def sweep(self, f: Features, lim: Limits, avoid: list[tuple] = (), budget_ms: float | None = None,
              all_variants: bool = False) -> list[tuple[float, int, int, int]]:
        """Coarse sweep over the next slice of variants. Returns non-overlapping
        candidates as (score, variant, fine cx, fine cy), best first."""
        t = self.templates
        budget = self.budget_ms if budget_ms is None else budget_ms
        if all_variants:
            ids = np.arange(t.n_variants, dtype=np.int64)
        else:
            ids = self._next_ids(self.chunk)
        c = t.coarse
        out_cost = np.empty((len(ids), TOPK), np.float64)
        out_x = np.empty((len(ids), TOPK), np.int64)
        out_y = np.empty((len(ids), TOPK), np.int64)
        t0 = time.perf_counter()
        _sweep_kernel(f.dist_coarse.ravel(), self.Hc, self.Wc, f.integral, ids, c.start, c.count,
                      c.w, c.h, self._tlen, self._spen, self._off_c, self._pdx_c, self._pdy_c, self._pbin_c,
                      self.n_bins, self.stride, float(self.tau_c), lim.threshold + self.slack, self.early,
                      lim.clutter_penalty, lim.clutter_rho0, lim.clutter_max, self.cross_penalty,
                      self.cross_free, out_cost, out_x, out_y)
        ms = (time.perf_counter() - t0) * 1000
        self.last_sweep_ms = ms
        if not all_variants and ms > 0.05:
            r = len(ids) / ms
            self.rate = 0.8 * self.rate + 0.2 * r
            self.chunk = int(np.clip(self.rate * budget, 8, t.n_variants))

        jj, kk = np.nonzero(out_x >= 0)
        cands = []
        for j, k in zip(jj, kk):
            v = int(ids[j])
            fcx = 2 * (int(out_x[j, k]) + int(c.cx[v]))
            fcy = 2 * (int(out_y[j, k]) + int(c.cy[v]))
            cands.append((float(out_cost[j, k]), v, fcx, fcy))
        cands.sort()
        cands = cands[:200]
        kept: list[tuple[float, int, int, int]] = []
        boxes = list(avoid)
        for sc, v, fcx, fcy in cands:
            box = self.box(v, fcx, fcy)
            if any(iou(box, b) > 0.3 for b in boxes):
                continue
            kept.append((sc, v, fcx, fcy))
            boxes.append(box)
            if len(kept) >= self.max_candidates:
                break
        return kept

    # ----------------------------------------------------------------- refine
    def box(self, v: int, cx: int, cy: int) -> tuple[int, int, int, int]:
        fl = self.templates.fine
        return (cx - int(fl.cx[v]), cy - int(fl.cy[v]), int(fl.w[v]), int(fl.h[v]))

    def refine(self, f: Features, seeds: list[tuple[int, int, int]], radius: int | None = None,
               same_variant_only: bool = False) -> list[Detection]:
        """seeds: (variant, fine cx, fine cy). Searches each seed's neighbour
        variants within +/- radius px; returns one Detection per seed."""
        if not seeds:
            return []
        t = self.templates
        r = self.refine_radius if radius is None else radius
        cand_vars = np.full((len(seeds), 9), -1, np.int64)
        for i, (v, _, _) in enumerate(seeds):
            nb = [v] if same_variant_only else t.neighbors(v)
            cand_vars[i, :len(nb)] = nb
        cxs = np.array([s[1] for s in seeds], np.int64)
        cys = np.array([s[2] for s in seeds], np.int64)
        out = np.empty((len(seeds), 9), np.float64)
        fl = t.fine
        t0 = time.perf_counter()
        _refine_kernel(f.dist.ravel(), self.Hf, self.Wf, f.integral, cand_vars, cxs, cys, r,
                       fl.start, fl.count, fl.w, fl.h, fl.cx, fl.cy, fl.binmask, self._tlen, self._spen, self._off_f,
                       self._pbin_f, self.n_bins, float(self.tau), self.delta,
                       self._lam, self._rho0, self.cross_penalty, self.cross_free, out)
        self.last_refine_ms = (time.perf_counter() - t0) * 1000
        dets = []
        for row in out:
            v = int(row[0])
            cx, cy = int(row[1]), int(row[2])
            box = self.box(v, cx, cy) if v >= 0 else (0, 0, 0, 0)
            dets.append(Detection(v, cx, cy, float(row[3]), float(row[4]), float(row[5]),
                                  float(row[6]), float(row[7]), box, float(row[8])))
        return dets

    def set_clutter(self, lim: Limits) -> None:
        self._lam = float(lim.clutter_penalty)
        self._rho0 = float(lim.clutter_rho0)

    def detect(self, f: Features, lim: Limits, avoid: list[tuple] = ()) -> list[Detection]:
        """One frame of global search: sweep a slice, refine, keep accepted."""
        self.set_clutter(lim)
        cands = self.sweep(f, lim, avoid)
        dets = self.refine(f, [(v, cx, cy) for _, v, cx, cy in cands])
        return [d for d in dets if d.accepted(lim)]


def warmup() -> None:
    """Compile the kernels (cached on disk after the first run)."""
    pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], np.int16)
    one = np.array([0], np.int64)
    i32 = np.array([4], np.int32)
    oc = np.empty((1, TOPK))
    ox = np.empty((1, TOPK), np.int64)
    D = np.full((8, 20, 30), 4, np.uint8)
    _sweep_kernel(D.ravel(), 20, 30, np.zeros((41, 61), np.int32), one, one, i32, i32, i32,
                  np.array([10.0]), np.array([0.0]), flat_offsets(pts, 20, 30), pts[:, 0].astype(np.int64), pts[:, 1].astype(np.int64),
                  pts[:, 2].astype(np.int64), 8, 2, 4.0, 0.5, 0.25, 1.0, 0.08, 0.3, 1.0, 0.08, oc, ox, ox.copy())
    out = np.empty((1, 9))
    Df = np.full((8, 40, 60), 4, np.uint8)
    _refine_kernel(Df.ravel(), 40, 60, np.zeros((41, 61), np.int32), np.array([[0]], np.int64),
                   np.array([10], np.int64), np.array([10], np.int64), 1, one, i32, i32, i32,
                   np.array([0], np.int32), np.array([0], np.int32), np.array([15], np.int32),
                   np.array([10.0]), np.array([0.0]), flat_offsets(pts, 40, 60), pts[:, 2].astype(np.int64), 8,
                   8.0, 2, 1.0, 0.08, 1.0, 0.08, out)
