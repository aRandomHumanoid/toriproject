"""Drawing: real edges near each sighting, glow, ghost strokes, debug view,
HUD and the object picker."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .edges import Features
from .geometry import Sketch, resample, transform, angle_bins
from .params import Config
from .tracker import Track

FONT = cv2.FONT_HERSHEY_SIMPLEX


def text_block(img: np.ndarray, lines: list[str], x: int, y: int, scale: float,
               color=(240, 240, 240), line_h: int | None = None) -> None:
    """Lines of text on a translucent dark box; y is the first baseline."""
    lines = [ln for ln in lines if ln]
    if not lines:
        return
    line_h = line_h or int(32 * scale) + 6
    widths = [cv2.getTextSize(ln, FONT, scale, 1)[0][0] for ln in lines]
    asc = cv2.getTextSize("Ag", FONT, scale, 1)[0][1]
    x0, y0 = max(0, x - 6), max(0, y - asc - 6)
    x1 = min(img.shape[1], x + max(widths) + 6)
    y1 = min(img.shape[0], y + line_h * (len(lines) - 1) + 8)
    roi = img[y0:y1, x0:x1]
    roi[:] = cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x, y + i * line_h), FONT, scale, color, 1, cv2.LINE_AA)


def _posed_points(track: Track, spacing: float, n_bins: int):
    """Smoothed pose of a track's sketch: per-stroke points (fine px) and bins."""
    x, y, scale, angle = track.pose
    sketch = track.templates.sketches[track.sketch_index]
    strokes = transform(sketch, scale, angle, track.mirrored, x, y)
    out = []
    for s in strokes:
        p, a = resample(s, spacing)
        out.append((p, angle_bins(a, n_bins)))
    return out


class Renderer:
    def __init__(self, cfg: Config):
        r, p = cfg.render, cfg.processing
        self.out_w, self.out_h = cfg.camera["width"], cfg.camera["height"]
        self.W, self.H = p["width"], p["height"]
        self.K = p["orientation_bins"]
        self.delta = p["delta"]
        self.sel_radius = 2
        self.min_support = 5
        rgb = r["line_color"]
        self.color = (rgb[2] / 255.0, rgb[1] / 255.0, rgb[0] / 255.0, 0.0)  # BGR scale
        self.line_width = max(1, int(r["line_width"]))
        self.glow = float(r["glow"])
        self.glow_strength = float(r["glow_strength"])
        self.ghost_opacity = float(r["ghost_opacity"])
        self.ghost_width = max(1, int(r["ghost_width"]))
        self.ghost_on = self.ghost_opacity > 0
        self.dim = float(r["background_dim"])
        self.desat = float(r["background_desaturate"])
        self.hud_seconds = float(r["hud_seconds"])
        fine_w = max(1, round(self.line_width / 2))
        self._kernel_fine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fine_w, fine_w))
        self.hud_pinned = False
        self._hud_until = 0.0

    # ---------------------------------------------------------------- selection
    def select(self, f: Features, track: Track):
        """Real edge pixels near the posed outline with a matching direction
        (support mask in a fine-res ROI), and the outline's unsupported runs."""
        r = self.sel_radius
        strokes = _posed_points(track, 2.0, self.K)
        allp = np.concatenate([p for p, _ in strokes], axis=0)
        x0 = int(max(0, np.floor(allp[:, 0].min()) - r - 1))
        y0 = int(max(0, np.floor(allp[:, 1].min()) - r - 1))
        x1 = int(min(self.W, np.ceil(allp[:, 0].max()) + r + 2))
        y1 = int(min(self.H, np.ceil(allp[:, 1].max()) + r + 2))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        tm = np.zeros((self.K, y1 - y0, x1 - x0), np.uint8)
        gaps = []
        for p, b in strokes:
            q = np.round(p - [x0, y0]).astype(np.int32)
            for i in range(len(q) - 1):
                cv2.line(tm[int(b[i])], (int(q[i, 0]), int(q[i, 1])),
                         (int(q[i + 1, 0]), int(q[i + 1, 1])), 1, 2 * r + 1)
            # A point is unsupported when no edge of its direction is within delta.
            pi = np.round(p).astype(np.int64)
            inside = (pi[:, 0] >= 0) & (pi[:, 0] < self.W) & (pi[:, 1] >= 0) & (pi[:, 1] < self.H)
            d = np.full(len(p), 255, np.uint8)
            d[inside] = f.dist[b[inside], pi[inside, 1], pi[inside, 0]]
            unsup = d > self.delta
            run = []
            for i in range(len(p)):
                if unsup[i]:
                    run.append(p[i])
                else:
                    if len(run) >= 2:
                        gaps.append(np.array(run))
                    run = []
            if len(run) >= 2:
                gaps.append(np.array(run))
        support = np.any(tm & f.soft[:, y0:y1, x0:x1], axis=0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
        if n > 1:
            small = stats[:, cv2.CC_STAT_AREA] < self.min_support
            small[0] = False
            support[small[labels]] = 0
        return (x0, y0, x1, y1), support, gaps

    # ------------------------------------------------------------------ overlay
    def render(self, frame: np.ndarray, f: Features, tracks: list[Track]) -> np.ndarray:
        """frame: camera image at output size (BGR)."""
        base = frame
        if self.desat > 0:
            g = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            base = cv2.addWeighted(base, 1 - self.desat, g, self.desat, 0)
        if self.dim > 0:
            base = cv2.convertScaleAbs(base, alpha=1 - self.dim)
        if not tracks:
            return base

        lines = np.zeros((self.H, self.W), np.float32)
        ghost = np.zeros((self.out_h, self.out_w), np.float32)
        sx, sy = self.out_w / self.W, self.out_h / self.H
        for t in tracks:
            sel = self.select(f, t)
            if sel is None:
                continue
            (x0, y0, x1, y1), support, gaps = sel
            roi = lines[y0:y1, x0:x1]
            np.maximum(roi, support.astype(np.float32) * t.opacity, out=roi)
            if self.ghost_on and gaps:
                pts = [np.round(gp * [sx, sy]).astype(np.int32) for gp in gaps]
                cv2.polylines(ghost, pts, False, float(self.ghost_opacity * t.opacity), self.ghost_width,
                              cv2.LINE_AA)

        # Lines and glow are combined at processing resolution and upscaled once:
        # one fine pixel becomes a soft 2 px line at output size.
        # Soften stair steps before upscaling; the gain restores a 1 px line's peak.
        layer = cv2.GaussianBlur(lines, (0, 0), 0.6) * 1.6
        if self.line_width > 2:
            layer = cv2.dilate(layer, self._kernel_fine)
        if self.glow > 0 and self.glow_strength > 0:
            glow = cv2.GaussianBlur(cv2.dilate(lines, np.ones((3, 3), np.uint8)), (0, 0), self.glow / 2)
            layer = layer + glow * (1.5 * self.glow_strength)
        up = cv2.resize(layer, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        up = cv2.min(up + ghost, 1.0)
        over = (up * 255).astype(np.uint8)
        b, g, r, _ = self.color
        if b == g == r == 1.0:
            over3 = cv2.cvtColor(over, cv2.COLOR_GRAY2BGR)
        else:
            over3 = cv2.merge([cv2.convertScaleAbs(over, alpha=b), cv2.convertScaleAbs(over, alpha=g),
                               cv2.convertScaleAbs(over, alpha=r)])
        return cv2.add(base, over3)

    # ---------------------------------------------------------------------- HUD
    def poke_hud(self) -> None:
        self._hud_until = time.monotonic() + self.hud_seconds

    def hud(self, img: np.ndarray, lines: list[str]) -> np.ndarray:
        if not (self.hud_pinned or time.monotonic() < self._hud_until):
            return img
        h = img.shape[0]
        text_block(img, lines, 18, h - 18 - 24 * (len(lines) - 1), 0.6, line_h=24)
        return img

    # -------------------------------------------------------------------- debug
    def debug(self, small: np.ndarray, f: Features, out: np.ndarray, tracks: list[Track],
              candidates: list, text: list[str]) -> np.ndarray:
        """2x2 grid: camera with boxes, edges by direction, distance map, output."""
        cam = small.copy()
        for c in candidates:
            x, y, w, h = c
            cv2.rectangle(cam, (x, y), (x + w, y + h), (0, 140, 255), 1)
        for t in tracks:
            x, y, w, h = t.box
            col = (80, 255, 80) if t.confirmed else (0, 220, 255)
            cv2.rectangle(cam, (x, y), (x + w, y + h), col, 2 if t.confirmed else 1)
        hue = np.where(f.bins >= 0, (f.bins.astype(np.int32) * 180 // self.K), 0).astype(np.uint8)
        hsv = cv2.merge([hue, np.full_like(hue, 220), (f.edges * 255).astype(np.uint8)])
        edges = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        dmin = f.dist.min(axis=0).astype(np.float32)
        dvis = cv2.applyColorMap((255 - dmin * (255.0 / max(dmin.max(), 1))).astype(np.uint8), cv2.COLORMAP_MAGMA)
        outs = cv2.resize(out, (self.W, self.H), interpolation=cv2.INTER_AREA)
        grid = np.vstack([np.hstack([cam, edges]), np.hstack([dvis, outs])])
        grid = cv2.resize(grid, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        text_block(grid, text, 14, 28, 0.55, line_h=22)
        return grid


class Picker:
    """Grid of objects to choose from (key c). Arrow keys or mouse to choose,
    Enter to confirm, Esc to cancel."""

    def __init__(self, names: list[str], previews: list[Sketch | None], current: int):
        self.names = names
        self.previews = previews
        self.sel = current
        self.cols = 5
        self.rows = (len(names) + self.cols - 1) // self.cols
        self.rects: list[tuple[int, int, int, int]] = []

    def move(self, dx: int, dy: int) -> None:
        n = len(self.names)
        i = self.sel + dx + dy * self.cols
        if 0 <= i < n:
            self.sel = i

    def hit(self, x: int, y: int) -> int | None:
        for i, (rx, ry, rw, rh) in enumerate(self.rects):
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return i
        return None

    def draw(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        out = cv2.convertScaleAbs(img, alpha=0.25)
        margin = 40
        tw = (w - 2 * margin) // self.cols
        th = min((h - 2 * margin - 30) // self.rows, tw)
        top = margin + 30 + ((h - 2 * margin - 30) - th * self.rows) // 2
        cv2.putText(out, "Choose an object  (arrows/mouse, Enter, Esc)", (margin, margin + 6), FONT, 0.7,
                    (230, 230, 230), 1, cv2.LINE_AA)
        self.rects = []
        for i, name in enumerate(self.names):
            r, c = divmod(i, self.cols)
            x, y = margin + c * tw, top + r * th
            self.rects.append((x, y, tw, th))
            pad = 8
            sel = i == self.sel
            cv2.rectangle(out, (x + pad, y + pad), (x + tw - pad, y + th - pad),
                          (255, 255, 255) if sel else (90, 90, 90), 2 if sel else 1)
            sk = self.previews[i]
            if sk is not None:
                size = min(tw, th) * 0.55
                cx, cy = x + tw / 2, y + th / 2 - 8
                for s in sk:
                    p = np.round(s * size + [cx, cy]).astype(np.int32)
                    cv2.polylines(out, [p], False, (255, 255, 255) if sel else (190, 190, 190),
                                  2, cv2.LINE_AA)
            (tw_text, _), _ = cv2.getTextSize(name, FONT, 0.55, 1)
            cv2.putText(out, name, (int(x + tw / 2 - tw_text / 2), y + th - pad - 10), FONT, 0.55,
                        (255, 255, 255) if sel else (170, 170, 170), 1, cv2.LINE_AA)
        return out
