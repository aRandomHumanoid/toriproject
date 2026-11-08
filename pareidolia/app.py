"""Main loop: camera in, overlay out.

    uv run pareidolia                       # webcam, object from config.toml
    uv run pareidolia --object hand
    uv run pareidolia --source clip.mp4     # a video file (loops)
    uv run pareidolia --source photo.jpg    # a slow pan over a still image

Keys: c picker, [ ] previous/next object, s shuffle, + - suggestibility,
g ghost strokes, d debug view, h pin the HUD, f fullscreen, q or Esc quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pareidolia  # noqa: F401  (sets threading and Qt environment first)
import cv2
import numpy as np

from .capture import open_source
from .catalog import Catalog, load_catalog
from .edges import EdgeProcessor
from .matcher import Matcher, warmup
from .params import Config, limits_for, load_config
from .render import Picker, Renderer
from .templates import TemplateStore
from .tracker import Tracker

WIN = "pareidolia"

# Arrow keys from cv2.waitKeyEx: GTK/X11 codes and Qt codes.
LEFT = {65361, 0x1000012, 2424832}
UP = {65362, 0x1000013, 2490368}
RIGHT = {65363, 0x1000014, 2555904}
DOWN = {65364, 0x1000015, 2621440}
ENTER = {13, 10, 65293, 0x1000004, 0x1000005}
ESC = {27, 0x1000000}


class Pipeline:
    """One frame end to end: edges, track updates, global search, rendering."""

    def __init__(self, cfg: Config, store: TemplateStore, catalog: Catalog, adaptive: bool = True):
        self.cfg = cfg
        self.store = store
        self.catalog = catalog
        self.edges = EdgeProcessor(cfg)
        self.matcher = Matcher(cfg)
        self.tracker = Tracker(cfg)
        self.renderer = Renderer(cfg)
        self.size = (cfg.processing["width"], cfg.processing["height"])
        self.suggestibility = float(cfg.match["suggestibility"])
        self.adaptive = adaptive
        self.frame_target_ms = 1000.0 / float(cfg.camera["fps"])
        self.max_budget = float(cfg.search["budget_ms"]) * 1.5
        self.budget = float(cfg.search["budget_ms"])
        self.timing: dict[str, float] = {}
        self.prev = None
        self.last_now: float | None = None
        self.candidates: list[tuple[int, int, int, int]] = []
        self.small: np.ndarray | None = None
        self.features = None
        self.templates = None
        self.set_object(catalog.active.name, fade=False)

    def set_object(self, name: str, fade: bool = True) -> None:
        self.templates = self.store.get(name)
        self.matcher.set_templates(self.templates)
        if fade:
            self.tracker.retire_all()
        else:
            self.tracker.reset()

    def limits(self):
        return limits_for(self.cfg, self.suggestibility, self.catalog.active.threshold_offset)

    def _time(self, key: str, t0: float) -> float:
        now = time.perf_counter()
        ms = (now - t0) * 1000
        self.timing[key] = 0.9 * self.timing.get(key, ms) + 0.1 * ms
        return now

    def step(self, frame: np.ndarray, now: float | None = None, render: bool = True) -> np.ndarray | None:
        now = time.monotonic() if now is None else now
        dt = 0.0 if self.last_now is None else max(now - self.last_now, 1e-3)
        self.last_now = now
        t_start = t = time.perf_counter()

        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        f = self.edges.process(small)
        t = self._time("edges", t)

        lim = self.limits()
        self.matcher.set_clutter(lim)
        self.tracker.update(self.prev, f, self.matcher, lim, now, dt)
        t = self._time("track", t)

        if self.adaptive:
            other = sum(v for k, v in self.timing.items() if k not in ("sweep", "frame"))
            self.budget = float(np.clip(self.frame_target_ms - other - 3.0, 3.0, self.max_budget))
        cands = self.matcher.sweep(f, lim, avoid=self.tracker.boxes(), budget_ms=self.budget)
        t = self._time("sweep", t)
        dets = self.matcher.refine(f, [(v, cx, cy) for _, v, cx, cy in cands])
        self.candidates = [d.box for d in dets]
        self.tracker.add([d for d in dets if d.accepted(lim)], self.templates, now)
        t = self._time("refine", t)

        out = None
        if render:
            out = self.renderer.render(frame, f, self.tracker.visible())
            t = self._time("render", t)
        self._time("frame", t_start)
        self.prev, self.small, self.features = f, small, f
        return out

    def debug_text(self, fps: float) -> list[str]:
        tm = self.timing
        lim = self.limits()
        period = self.matcher.sweep_period_s
        return [
            f"{self.catalog.active.name}  s={self.suggestibility:.2f}  theta={lim.threshold:.2f} "
            f"cov>={lim.coverage:.2f} rho<={lim.clutter_max:.2f}  fps {fps:.1f}",
            f"edges {tm.get('edges', 0):.1f}  track {tm.get('track', 0):.1f}  sweep {tm.get('sweep', 0):.1f}"
            f" (budget {self.budget:.1f})  refine {tm.get('refine', 0):.1f}  render {tm.get('render', 0):.1f} ms",
            f"variants {self.templates.n_variants}  chunk {self.matcher.chunk}  full sweep every "
            f"{period:.2f}s" if period else f"variants {self.templates.n_variants}  chunk {self.matcher.chunk}",
            f"tracks {len(self.tracker.tracks)} (shown {len(self.tracker.visible())})  "
            f"sightings {self.tracker.sightings}  edge density {self.features.density:.3f}"
            if self.features else "",
        ]


def build(args) -> tuple[Config, Catalog, TemplateStore]:
    cfg = load_config(args.config)
    if args.suggestibility is not None:
        cfg.match["suggestibility"] = args.suggestibility
    cat_file = load_catalog()
    store = TemplateStore(cat_file.objects, cfg, first=args.object or cfg.catalog["object"])
    if store.missing:
        print(f"note: no templates yet for {', '.join(store.missing)}; run "
              f"`uv run python tools/build_templates.py --catalog-only`", file=sys.stderr)
    objects = [o for o in cat_file.objects if store.available(o.name)]
    if not objects:
        sys.exit("no templates built. Run: uv run python tools/build_templates.py --catalog-only")
    catalog = Catalog(objects, start=args.object or cfg.catalog["object"],
                      shuffle_minutes=float(cfg.catalog["shuffle_minutes"]))
    return cfg, catalog, store


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="camera index, video file or image (default: camera from config)")
    ap.add_argument("--object", help="object to start with")
    ap.add_argument("--config", help="config file (default: config.toml)")
    ap.add_argument("--suggestibility", type=float)
    ap.add_argument("--debug", action="store_true", help="start in the debug view")
    ap.add_argument("--headless", action="store_true", help="no window (for testing)")
    ap.add_argument("--frames", type=int, default=0, help="stop after this many frames")
    ap.add_argument("--save", help="directory to save output frames into")
    ap.add_argument("--save-every", type=int, default=30)
    args = ap.parse_args(argv)

    cv2.setNumThreads(4)
    cfg, catalog, store = build(args)
    print("compiling kernels (cached after the first run)...", flush=True)
    warmup()
    source = open_source(args.source, cfg, realtime=not args.headless)
    pipe = Pipeline(cfg, store, catalog)
    rend = pipe.renderer
    debug = args.debug
    picker: Picker | None = None
    fullscreen = bool(cfg.render["fullscreen"])
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    clicked: list[tuple[int, int]] = []
    if not args.headless:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, cfg.camera["width"], cfg.camera["height"])
        if fullscreen:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(WIN, lambda ev, x, y, *_: clicked.append((x, y))
                             if ev == cv2.EVENT_LBUTTONDOWN else None)
    rend.poke_hud()

    def switch(index: int) -> None:
        if catalog.select(index):
            pipe.set_object(catalog.active.name)
        rend.poke_hud()

    fps, last, n = 0.0, time.perf_counter(), 0
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            out = pipe.step(frame)
            n += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if n > 1 else 0.0
            last = now
            if catalog.maybe_shuffle():
                pipe.set_object(catalog.active.name)
                rend.poke_hud()

            if debug:
                out = rend.debug(pipe.small, pipe.features, out, pipe.tracker.tracks, pipe.candidates,
                                 pipe.debug_text(fps))
            if picker:
                out = picker.draw(out)
            else:
                lines = [f"{catalog.active.name}   suggestibility {pipe.suggestibility:.2f}   {fps:.0f} fps"
                         + ("   shuffle" if catalog.shuffle_on else "")]
                out = rend.hud(out, lines)

            if save_dir and n % args.save_every == 0:
                cv2.imwrite(str(save_dir / f"frame_{n:05d}.jpg"), out)
            if args.frames and n >= args.frames:
                break
            if args.headless:
                continue

            cv2.imshow(WIN, out)
            key = cv2.waitKeyEx(1)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if picker:
                if clicked:
                    hit = picker.hit(*clicked[-1])
                    clicked.clear()
                    if hit is not None:
                        switch(hit)
                        picker = None
                    continue
                if key in LEFT:
                    picker.move(-1, 0)
                elif key in RIGHT:
                    picker.move(1, 0)
                elif key in UP:
                    picker.move(0, -1)
                elif key in DOWN:
                    picker.move(0, 1)
                elif key in ENTER or key == ord(" "):
                    switch(picker.sel)
                    picker = None
                elif key in ESC or key == ord("c"):
                    picker = None
                continue
            clicked.clear()
            if key == -1:
                continue
            ch = chr(key & 0xFF) if key & 0xFF < 128 and key < 0x10000 else ""
            if key in ESC or ch == "q":
                break
            if ch == "c":
                picker = Picker(catalog.names(), [store.preview(nm) for nm in catalog.names()],
                                catalog.index)
            elif ch == "]":
                switch(catalog.index + 1)
            elif ch == "[":
                switch(catalog.index - 1)
            elif ch == "s":
                catalog.toggle_shuffle()
                rend.poke_hud()
            elif ch in ("+", "="):
                pipe.suggestibility = min(1.0, round(pipe.suggestibility + 0.05, 2))
                rend.poke_hud()
            elif ch in ("-", "_"):
                pipe.suggestibility = max(0.0, round(pipe.suggestibility - 0.05, 2))
                rend.poke_hud()
            elif ch == "g":
                rend.ghost_on = not rend.ghost_on
            elif ch == "d":
                debug = not debug
            elif ch == "h":
                rend.hud_pinned = not rend.hud_pinned
            elif ch == "f":
                fullscreen = not fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        store.close()
        if not args.headless:
            cv2.destroyAllWindows()
    if args.headless:
        tm = pipe.timing
        print(f"{n} frames  " + "  ".join(f"{k} {v:.1f}ms" for k, v in tm.items())
              + f"  sightings {pipe.tracker.sightings}")


if __name__ == "__main__":
    main()
