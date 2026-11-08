# Pareidolia overlay

A live webcam piece that finds outlines of a chosen object hiding in the scene
(a fish, a hand, a teapot...) and lights up the real edges that form them.
Matching is oriented chamfer matching against hand-drawn
[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) sketches;
no machine learning. Runs on the CPU at about 30 fps on a Ryzen AI 7 350.

## Run

```bash
uv sync                                  # once: Python 3.13 + OpenCV + NumPy + Numba
uv run pareidolia                        # webcam, object from config.toml
uv run pareidolia --object hand --debug  # start with a given object, debug view on
uv run pareidolia --source photo.jpg     # a slow pan over a still image (no camera)
uv run pareidolia --source clip.mp4      # a video file, looped
```

The first launch compiles the Numba kernels (about 10 s); later launches use the cache.

| Key | Action |
| --- | --- |
| `c` | Object picker (arrows or mouse, Enter to choose, Esc to cancel) |
| `[` `]` | Previous / next object |
| `s` | Shuffle: a random object every few minutes |
| `+` `-` | Suggestibility up / down |
| `g` | Ghost strokes (the drawn-in gaps) on / off |
| `d` | Debug view: candidates, edges by direction, distance map, timings |
| `h` | Keep the status line visible |
| `f` | Fullscreen |
| `q` / Esc | Quit |

## Objects

`catalog.toml` lists the 20 objects, each with its search rotation and a
calibrated threshold offset. Templates (40 curated sketches per object) live in
`templates/`. All 20 catalog objects and 12 reserves are already built.

```bash
uv run python tools/build_templates.py                 # (re)build catalog + reserves
uv run python tools/contact_sheet.py fish              # look at fish's 40 sketches
uv run python tools/build_templates.py fish --drop 3 17   # reject sketches 3 and 17, refill
```

## Calibrate for the space it will show in

Simple shapes turn up far more often than complex ones. Calibration evens this out
so suggestibility 0.5 gives about 3 sightings per minute for every object.

```bash
uv run python tools/record_clip.py --seconds 60        # record 2-3 clips in the real space and light
uv run python tools/calibrate.py                       # writes threshold offsets to catalog.toml
uv run python tools/calibrate.py --reserves            # also measures the reserves, for ranking
uv run python tools/rank_categories.py                 # rank all 32 candidates
uv run python tools/rank_categories.py --write         # keep the top 20 in catalog.toml
```

Calibration runs the real pipeline 5 times per object; with one 60 s clip expect
roughly 30-40 minutes for all 20 (use `--seconds 20 --step 3` for a quick pass).

## Tuning (`config.toml`)

- `match.suggestibility`: the main control (0 to 1). It moves the match threshold,
  the required outline coverage and the clutter limit together.
- `search.size_bias`, `search.scale_min/max`: how large sightings tend to be.
- `match.clutter_*`, `match.cross_*`: how strongly busy texture is rejected.
- `track.*`: how many sightings at once, how fast they fade, how long they live.
- `render.*`: line color, width, glow, ghost strokes, background dimming.

Changing scales, rotations or the point spacing rebuilds the variant cache on the
next launch (a few seconds; runs in the background for the non-active objects).

```bash
uv run python tools/bench.py                           # per-stage timing for every object
```

## Troubleshooting

- **Camera gives about 10 fps**: most laptop webcams lengthen the exposure in dim
  light. Add light, or install `v4l-utils` and run
  `v4l2-ctl -c exposure_dynamic_framerate=0` (control names vary by camera;
  `v4l2-ctl -l` lists them).
- **No window on Wayland**: the app runs OpenCV's Qt window through XWayland
  (`QT_QPA_PLATFORM=xcb`), which needs `DISPLAY` to be set.
- **Slow parallel code**: the package selects Numba's `workqueue` threading layer,
  because the OpenMP layer cost about 6 ms per call on this machine.

## How it works

1. Each frame is reduced to 640x360, edges found with Canny, and each edge pixel
   binned into one of 8 directions (with both neighbouring bins, for tolerance).
2. A capped distance-to-edge map is built per direction at 640x360 and 320x180.
3. A time-budgeted coarse sweep scores a slice of the object's template variants
   (sketch x scale x rotation x mirror) at every position, rejecting busy texture
   early. The best candidates are refined at full resolution, scored on distance,
   coverage, excess edge density and crossing edges.
4. A candidate becomes a sighting after 3 consecutive hits; it is then followed
   with optical flow, re-checked locally each frame, smoothed and faded.
5. Only real edge pixels near the matched outline (and with its direction) are
   drawn, with a glow; gaps can be filled with faint ghost strokes.

Code: `pareidolia/` (edges, matcher, tracker, render, templates, catalog, capture,
app) and `tools/` (build_templates, contact_sheet, record_clip, calibrate,
rank_categories, bench).

Sketches: Quick, Draw! dataset by Google, CC BY 4.0.
