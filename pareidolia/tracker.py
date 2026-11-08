"""Sightings over time: confirmation, optical-flow following, smoothing, fades."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .edges import Features
from .matcher import Detection, Matcher, iou
from .params import Config, Limits
from .templates import ObjectTemplates


class OneEuro:
    """One Euro filter: smooth when still, responsive when moving."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: float | None = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, dt: float) -> float:
        if self.x is None or dt <= 0:
            self.x = x
            return x
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


@dataclass
class Track:
    id: int
    templates: ObjectTemplates
    variant: int
    cx: float                # matched sketch center, fine px
    cy: float
    born: float
    det: Detection
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    confirmed_at: float = 0.0
    retiring: bool = False
    opacity: float = 0.0
    pose: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0)  # smoothed x, y, scale, angle
    filters: list[OneEuro] = field(default_factory=list)

    @property
    def sketch_index(self) -> int:
        return self.templates.unindex(self.variant)[0]

    @property
    def mirrored(self) -> bool:
        return self.templates.mirrors[self.templates.unindex(self.variant)[1]]

    @property
    def box(self) -> tuple[int, int, int, int]:
        fl = self.templates.fine
        v = self.variant
        return (int(self.cx) - int(fl.cx[v]), int(self.cy) - int(fl.cy[v]), int(fl.w[v]), int(fl.h[v]))

    @property
    def visible(self) -> bool:
        return self.confirmed and self.opacity > 0.005


class Tracker:
    def __init__(self, cfg: Config):
        t = cfg.track
        self.max_tracks = int(t["max_tracks"])
        self.max_tentative = int(t["max_tentative"])
        self.confirm_frames = int(t["confirm_frames"])
        self.max_misses = int(t["max_misses"])
        self.fade_in = float(t["fade_in_s"])
        self.fade_out = float(t["fade_out_s"])
        self.max_lifetime = float(t["max_lifetime_s"])
        self.cooldown = float(t["cooldown_s"])
        self.local_radius = int(t["local_radius"])
        self.hysteresis = float(t["hysteresis"])
        self.overlap = float(t["overlap_iou"])
        self.min_cutoff = float(t["smooth_min_cutoff"])
        self.beta = float(t["smooth_beta"])
        self.tracks: list[Track] = []
        self.cooldowns: list[tuple[tuple[int, int, int, int], float]] = []
        self.sightings = 0      # confirmed sightings so far (for calibration)
        self._next_id = 1

    # ------------------------------------------------------------------ helpers
    def boxes(self) -> list[tuple[int, int, int, int]]:
        return [t.box for t in self.tracks]

    def visible(self) -> list[Track]:
        return [t for t in self.tracks if t.visible]

    def n_confirmed(self) -> int:
        return sum(1 for t in self.tracks if t.confirmed and not t.retiring)

    def retire_all(self) -> None:
        """Fade everything out (object switch)."""
        for t in self.tracks:
            t.retiring = True

    def reset(self) -> None:
        self.tracks.clear()
        self.cooldowns.clear()

    def _smooth(self, t: Track, dt: float) -> None:
        _, _, scale, angle = t.templates.pose(t.variant)
        if not t.filters:
            t.filters = [OneEuro(self.min_cutoff, self.beta) for _ in range(4)]
            t.pose = (t.cx, t.cy, scale, angle)
        # Unwrap the angle against the smoothed one so 350 -> 10 does not spin.
        prev = t.pose[3]
        angle = prev + ((angle - prev + 180) % 360 - 180)
        vals = (t.cx, t.cy, scale, angle)
        t.pose = tuple(f(v, dt) for f, v in zip(t.filters, vals))

    @staticmethod
    def _flow(prev_gray: np.ndarray, gray: np.ndarray, box) -> tuple[float, float]:
        """Median optical flow of corner features inside a box."""
        H, W = gray.shape
        x, y, w, h = box
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return 0.0, 0.0
        p0 = cv2.goodFeaturesToTrack(prev_gray[y0:y1, x0:x1], maxCorners=30, qualityLevel=0.01,
                                     minDistance=5)
        if p0 is None or len(p0) < 3:
            return 0.0, 0.0
        p0 = p0.astype(np.float32) + np.array([x0, y0], np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(15, 15), maxLevel=2)
        ok = st.ravel() == 1
        if ok.sum() < 3:
            return 0.0, 0.0
        d = (p1 - p0).reshape(-1, 2)[ok]
        return float(np.median(d[:, 0])), float(np.median(d[:, 1]))

    # ------------------------------------------------------------------ update
    def update(self, prev: Features | None, f: Features, matcher: Matcher, lim: Limits,
               now: float, dt: float) -> None:
        """Follow, re-check, confirm, fade and drop existing tracks."""
        active = matcher.templates
        live = [t for t in self.tracks if t.templates is active and not t.retiring]
        flows = {}
        if prev is not None:
            for t in self.tracks:
                flows[t.id] = self._flow(prev.gray, f.gray, t.box)
        seeds = []
        for t in live:
            fx, fy = flows.get(t.id, (0.0, 0.0))
            seeds.append((t.variant, int(round(t.cx + fx)), int(round(t.cy + fy))))
        dets = matcher.refine(f, seeds, radius=self.local_radius) if seeds else []
        relaxed = lim.relaxed(self.hysteresis)
        for t, d in zip(live, dets):
            ok = d.accepted(relaxed if t.confirmed else lim)
            if ok:
                t.variant, t.cx, t.cy, t.det = d.variant, float(d.cx), float(d.cy), d
                t.hits += 1
                t.misses = 0
            else:
                fx, fy = flows.get(t.id, (0.0, 0.0))
                t.cx += fx
                t.cy += fy
                t.misses += 1
                t.hits = 0 if not t.confirmed else t.hits
        for t in self.tracks:
            if t not in live:  # retiring or from a previous object: just drift and fade
                fx, fy = flows.get(t.id, (0.0, 0.0))
                t.cx += fx
                t.cy += fy

        keep = []
        for t in self.tracks:
            if not t.confirmed:
                if t.misses > 0 or t.retiring:
                    continue  # a tentative sighting must hit every frame
                if t.hits >= self.confirm_frames and self.n_confirmed() < self.max_tracks:
                    t.confirmed = True
                    t.confirmed_at = now
                    self.sightings += 1
                keep.append(t)
                continue
            if self.max_lifetime > 0 and now - t.confirmed_at >= self.max_lifetime:
                t.retiring = True
            if t.retiring or t.misses > 0:
                t.opacity -= dt / max(self.fade_out, 1e-3)
            else:
                t.opacity += dt / max(self.fade_in, 1e-3)
            t.opacity = min(max(t.opacity, 0.0), 1.0)
            gone = t.opacity <= 0.0 and (t.retiring or t.misses > 0)
            if t.misses >= self.max_misses or gone:
                if t.retiring and self.cooldown > 0:
                    self.cooldowns.append((t.box, now + self.cooldown))
                continue
            self._smooth(t, dt)
            keep.append(t)
        self.tracks = keep
        self.cooldowns = [(b, until) for b, until in self.cooldowns if until > now]

    def add(self, dets: list[Detection], templates: ObjectTemplates, now: float) -> None:
        """Start tentative tracks for new detections that do not overlap anything."""
        for d in sorted(dets, key=lambda d: d.score):
            if sum(1 for t in self.tracks if not t.confirmed) >= self.max_tentative:
                break
            if any(iou(d.box, t.box) > self.overlap for t in self.tracks):
                continue
            if any(iou(d.box, b) > self.overlap for b, _ in self.cooldowns):
                continue
            t = Track(id=self._next_id, templates=templates, variant=d.variant, cx=float(d.cx),
                      cy=float(d.cy), born=now, det=d)
            self._next_id += 1
            self._smooth(t, 0.0)
            self.tracks.append(t)
