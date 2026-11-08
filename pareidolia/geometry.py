"""Stroke geometry shared by the offline template builder and the live pipeline.

A sketch is a list of strokes; a stroke is an (N, 2) float array of x, y points.
Normalized sketches are centered on their bounding box with the longest side = 1.
"""

from __future__ import annotations

import numpy as np

Sketch = list[np.ndarray]


def clean_stroke(stroke: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate points."""
    stroke = np.asarray(stroke, dtype=np.float64)
    if len(stroke) < 2:
        return stroke
    keep = np.ones(len(stroke), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(stroke, axis=0)) > 1e-9, axis=1)
    return stroke[keep]


def stroke_length(stroke: np.ndarray) -> float:
    if len(stroke) < 2:
        return 0.0
    d = np.diff(stroke, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def sketch_length(sketch: Sketch) -> float:
    return sum(stroke_length(s) for s in sketch)


def sketch_bbox(sketch: Sketch) -> tuple[float, float, float, float]:
    allp = np.concatenate(sketch, axis=0)
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def normalize(sketch: Sketch) -> Sketch:
    """Center on the bounding box and scale so the longest side is 1."""
    x0, y0, x1, y1 = sketch_bbox(sketch)
    side = max(x1 - x0, y1 - y0, 1e-9)
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return [(s - c) / side for s in sketch]


def resample(stroke: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Points at even arc-length spacing along a polyline, and the tangent angle
    at each point in degrees modulo 180 (image coordinates, y down)."""
    stroke = clean_stroke(stroke)
    if len(stroke) < 2:
        return stroke[:1].copy(), np.zeros(min(len(stroke), 1))
    d = np.diff(stroke, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(int(round(total / spacing)), 1)
    t = np.linspace(0.0, total, n + 1)
    idx = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seg) - 1)
    frac = (t - cum[idx]) / np.maximum(seg[idx], 1e-12)
    pts = stroke[idx] + d[idx] * frac[:, None]
    ang = np.degrees(np.arctan2(d[idx, 1], d[idx, 0])) % 180.0
    return pts, ang


def transform(sketch: Sketch, scale: float, angle_deg: float, mirror: bool,
              cx: float = 0.0, cy: float = 0.0) -> Sketch:
    """Scale (longest side in px), rotate (degrees, clockwise on screen), mirror
    horizontally, then translate a normalized sketch."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    rot = np.array([[ca, -sa], [sa, ca]])
    out = []
    for s in sketch:
        p = s.copy()
        if mirror:
            p[:, 0] = -p[:, 0]
        p = (p * scale) @ rot.T
        p[:, 0] += cx
        p[:, 1] += cy
        out.append(p)
    return out


def angle_bins(angles_deg: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantize line directions (mod 180) into n_bins orientation bins."""
    return (np.floor((angles_deg % 180.0) / (180.0 / n_bins)).astype(np.int64)) % n_bins


def sample_points(sketch: Sketch, spacing: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample every stroke; return float points (M, 2) and orientation bins (M,)."""
    pts, angs = [], []
    for s in sketch:
        p, a = resample(s, spacing)
        pts.append(p)
        angs.append(a)
    P = np.concatenate(pts, axis=0)
    A = np.concatenate(angs, axis=0)
    return P, angle_bins(A, n_bins)


def sketch_from_quickdraw(drawing: list) -> Sketch:
    """Quick, Draw! simplified format: [[xs], [ys]] per stroke."""
    return [clean_stroke(np.stack([np.asarray(st[0], float), np.asarray(st[1], float)], axis=1))
            for st in drawing]


def pack_sketches(sketches: list[Sketch]) -> dict[str, np.ndarray]:
    """Flatten sketches into arrays for np.savez."""
    points, stroke_len, sketch_nstrokes = [], [], []
    for sk in sketches:
        sketch_nstrokes.append(len(sk))
        for s in sk:
            points.append(np.asarray(s, np.float32))
            stroke_len.append(len(s))
    return {
        "points": np.concatenate(points, axis=0) if points else np.zeros((0, 2), np.float32),
        "stroke_len": np.asarray(stroke_len, np.int32),
        "sketch_nstrokes": np.asarray(sketch_nstrokes, np.int32),
    }


def unpack_sketches(data) -> list[Sketch]:
    points = np.asarray(data["points"], np.float64)
    stroke_len = data["stroke_len"]
    nstrokes = data["sketch_nstrokes"]
    sketches, p, k = [], 0, 0
    for ns in nstrokes:
        sk = []
        for _ in range(int(ns)):
            n = int(stroke_len[k])
            sk.append(points[p:p + n])
            p += n
            k += 1
        sketches.append(sk)
    return sketches
