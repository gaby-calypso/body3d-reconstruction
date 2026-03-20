"""
main.py
-------
Entry point for the 3D body silhouette reconstruction pipeline.

Usage:
    python main.py

Each step of the pipeline is called from here in sequence.
Future steps are commented out and will be activated incrementally.
"""

from pathlib import Path
from src.loader import load_frame


# ── Configuration ─────────────────────────────────────────────────────────────
# Change these paths to point to your actual data files when using the camera.
RGB_PATH   = Path("data/sample/rgb.png")
DEPTH_PATH = Path("data/sample/depth.npy")


def main() -> None:
    print("=" * 50)
    print("  3D Body Reconstruction Pipeline")
    print("=" * 50)

    # ── Step 1: Data Loading ───────────────────────────────────────────────────
    print("\n[Step 1] Loading input data...")
    frame = load_frame(RGB_PATH, DEPTH_PATH)

    rgb   = frame["rgb"]
    depth = frame["depth"]

    print(f"  ✓ RGB image loaded:  shape={rgb.shape},   dtype={rgb.dtype}")
    print(f"  ✓ Depth map loaded:  shape={depth.shape}, dtype={depth.dtype}")
    print(f"  ✓ Depth range:       min={depth.min():.1f} mm, "
          f"max={depth.max():.1f} mm")

    # ── Step 2: Preprocessing (next session) ──────────────────────────────────
    # from src.preprocessing import preprocess_frame
    # frame = preprocess_frame(frame)

    # ── Step 3: Segmentation (coming later) ───────────────────────────────────
    # from src.segmentation import segment_body
    # mask = segment_body(frame)

    print("\n[Done] Pipeline complete for today's step.")


if __name__ == "__main__":
    main()