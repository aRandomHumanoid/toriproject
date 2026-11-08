"""Edge extraction and oriented distance transforms."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numba import njit, prange

from .params import Config


@dataclass
class Features:
    gray: np.ndarray         # (H, W) uint8, blurred
    edges: np.ndarray        # (H, W) uint8 0/1
    bins: np.ndarray         # (H, W) int8 orientation bin per edge pixel, -1 elsewhere
    soft: np.ndarray         # (K, H, W) uint8 0/1: edge pixels in bin k or a neighbour bin
    dist: np.ndarray         # (K, H, W) uint8 distance to nearest soft edge, capped at tau
    dist_coarse: np.ndarray  # (K, H/2, W/2) uint8, capped at tau_coarse
    integral: np.ndarray     # (H+1, W+1) int32 integral image of edges
    density: float           # edge pixels / frame pixels


@njit(parallel=True, cache=True, fastmath=True)
def _orient(edges, gx, gy, n_bins, bins, soft):
    """Edge direction (gradient + 90 deg, mod 180) -> bin; soft[k] marks edge
    pixels in bin k and both neighbouring bins."""
    H, W = edges.shape
    width = np.pi / n_bins
    for y in prange(H):
        for x in range(W):
            if edges[y, x] == 0:
                bins[y, x] = -1
                continue
            a = np.arctan2(gy[y, x], gx[y, x]) + 0.5 * np.pi
            a = a % np.pi
            b = int(a / width) % n_bins
            bins[y, x] = b
            soft[b, y, x] = 1
            soft[(b + 1) % n_bins, y, x] = 1
            soft[(b - 1 + n_bins) % n_bins, y, x] = 1


@njit(parallel=True, cache=True, fastmath=True)
def _pool2(soft, out):
    """2x2 max-pool of every bin."""
    K, H, W = out.shape
    for k in prange(K):
        for y in range(H):
            for x in range(W):
                out[k, y, x] = soft[k, 2 * y, 2 * x] | soft[k, 2 * y, 2 * x + 1] | \
                    soft[k, 2 * y + 1, 2 * x] | soft[k, 2 * y + 1, 2 * x + 1]


@njit(parallel=True, cache=True, fastmath=True)
def _chamfer_dt(soft, cap, out, scratch):
    """Distance to the nearest edge per bin, capped at `cap` px, as uint8.

    Two-pass 3-4 integer chamfer (distances in thirds of a pixel, within ~8% of
    Euclidean). In each row the previous-row terms are a vectorizable pass and
    only the left/right scan is sequential."""
    K, H, W = soft.shape
    C = 3 * cap
    for k in prange(K):
        d = scratch[k]
        for y in range(H):
            d[y, 0] = C
            d[y, W + 1] = C
            for x in range(W):
                d[y, x + 1] = 0 if soft[k, y, x] else C
            if y > 0:
                for x in range(1, W + 1):
                    v = d[y, x]
                    v = min(v, d[y - 1, x] + 3)
                    v = min(v, d[y - 1, x - 1] + 4)
                    v = min(v, d[y - 1, x + 1] + 4)
                    d[y, x] = v
            for x in range(1, W + 1):
                v = d[y, x - 1] + 3
                if v < d[y, x]:
                    d[y, x] = v
        for y in range(H - 1, -1, -1):
            if y < H - 1:
                for x in range(1, W + 1):
                    v = d[y, x]
                    v = min(v, d[y + 1, x] + 3)
                    v = min(v, d[y + 1, x + 1] + 4)
                    v = min(v, d[y + 1, x - 1] + 4)
                    d[y, x] = v
            for x in range(W, 0, -1):
                v = d[y, x + 1] + 3
                if v < d[y, x]:
                    d[y, x] = v
            for x in range(W):
                q = (d[y, x + 1] + 1) // 3
                out[k, y, x] = q if q < cap else cap


class _Buffers:
    def __init__(self, K: int, H: int, W: int):
        self.bins = np.empty((H, W), np.int8)
        self.soft = np.zeros((K, H, W), np.uint8)
        self.dist = np.empty((K, H, W), np.uint8)
        self.pooled = np.empty((K, H // 2, W // 2), np.uint8)
        self.dist_c = np.empty((K, H // 2, W // 2), np.uint8)


class EdgeProcessor:
    """Turns a 640x360 BGR frame into edge features. Output arrays come from two
    alternating buffer sets, so the previous frame's features stay valid for one
    more frame without allocating every frame."""

    def __init__(self, cfg: Config):
        p = cfg.processing
        self.W, self.H = p["width"], p["height"]
        self.K = p["orientation_bins"]
        self.tau = int(p["tau"])
        self.tau_c = int(p["tau_coarse"])
        self.sigma = float(p["blur_sigma"])
        self.min_fragment = int(p["min_fragment"])
        self._bufs = [_Buffers(self.K, self.H, self.W) for _ in range(2)]
        self._scratch = np.empty((self.K, self.H, self.W + 2), np.int16)
        self._scratch_c = np.empty((self.K, self.H // 2, self.W // 2 + 2), np.int16)
        self._i = 0

    def process(self, bgr_small: np.ndarray) -> Features:
        buf = self._bufs[self._i]
        self._i ^= 1
        gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), self.sigma)

        # Canny with thresholds from the median brightness, clamped for dark/bright scenes.
        med = float(np.median(gray[::4, ::4]))
        lo = float(np.clip(0.66 * med, 20, 110))
        hi = float(np.clip(1.33 * med, lo + 30, 230))
        edges = cv2.Canny(gray, lo, hi, L2gradient=True)

        # Drop short fragments (sensor noise, texture specks).
        _, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < self.min_fragment
        small[0] = False
        if small.any():
            edges[small[labels]] = 0
        edges = (edges > 0).astype(np.uint8)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        buf.soft.fill(0)
        _orient(edges, gx, gy, self.K, buf.bins, buf.soft)
        _chamfer_dt(buf.soft, self.tau, buf.dist, self._scratch)
        _pool2(buf.soft, buf.pooled)
        _chamfer_dt(buf.pooled, self.tau_c, buf.dist_c, self._scratch_c)

        integral = cv2.integral(edges, sdepth=cv2.CV_32S)
        return Features(gray=gray, edges=edges, bins=buf.bins, soft=buf.soft, dist=buf.dist,
                        dist_coarse=buf.dist_c, integral=integral,
                        density=float(integral[-1, -1]) / float(self.W * self.H))
