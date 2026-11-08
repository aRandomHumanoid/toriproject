"""Record reference clips from the camera for calibration and benchmarking.

    uv run python tools/record_clip.py                 # 60 s with a preview window
    uv run python tools/record_clip.py --seconds 120 --no-preview

Clips are written to clips/clip_<date>_<time>.mp4. Record the space the piece
will show, under its real lighting, with some movement in frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pareidolia  # noqa: E402,F401
import cv2  # noqa: E402

from pareidolia.capture import Camera  # noqa: E402
from pareidolia.params import ROOT, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--device", type=int)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    cam = Camera(cfg, args.device)
    W, H, fps = cfg.camera["width"], cfg.camera["height"], int(cfg.camera["fps"])
    out_dir = ROOT / "clips"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / time.strftime("clip_%Y%m%d_%H%M%S.mp4")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    print(f"recording {args.seconds:.0f} s to {path} (q to stop early)")
    start = time.monotonic()
    n = 0
    try:
        while time.monotonic() - start < args.seconds:
            frame = cam.read()
            if frame is None:
                continue
            writer.write(frame)
            n += 1
            if not args.no_preview:
                cv2.imshow("recording", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        writer.release()
        cam.close()
        cv2.destroyAllWindows()
    print(f"{n} frames, {n / max(time.monotonic() - start, 1e-6):.1f} fps")


if __name__ == "__main__":
    main()
