"""Pareidolia overlay: finds hidden outlines of a chosen object in live video."""

import os

# Numba's OpenMP layer costs ~6 ms per parallel call on this machine; the
# workqueue layer costs ~0.1 ms. Set before numba is imported anywhere.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# OpenCV's bundled Qt only ships the X11 plugin; on Wayland it runs via XWayland.
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
