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

    # ── Step 2: Preprocessing ──────────────────────────────────
    print("\n[Step 2] Preprocesando mapa de profundidad...")
    from src.preprocessing import preprocess_depth
    depth_clean = preprocess_depth(depth)
    print(f"✓ Depth preprocesado: shape={depth_clean.shape}, "
          f"dtype={depth_clean.dtype}")

    # ── Step 3: Segmentation (coming later) ───────────────────────────────────
    print("\n[Step 3] Segmentando silueta corporal...")
    from src.segmentation import segment_body
    seg = segment_body(rgb, depth_clean)

    print(f"  ✓ Máscara generada:  shape={seg['mask'].shape}, "
          f"dtype={seg['mask'].dtype}")
    print(f"  ✓ Píxeles del cuerpo: {seg['body_pixels']:,}")

    # ── Step 4: 3D Reconstruction ─────────────────────────────────────────────
    print("\n[Step 4] Reconstruyendo nube de puntos 3D...")
    from src.reconstruction import reconstruct_pointcloud
    pcd = reconstruct_pointcloud(seg["depth_body"], seg["rgb_body"])
    print(f"  ✓ Nube de puntos lista: {len(pcd.points):,} puntos 3D")

    # ── Step 5: Measurements ──────────────────────────────────────────────────
    print("\n[Step 5] Extrayendo medidas antropométricas...")
    from src.measurements import extract_measurements
    measurements = extract_measurements(seg["mask"], seg["depth_body"])

    print("\n  Resumen de medidas:")
    print("  " + "-" * 40)
    for zone, data in measurements.items():
        print(f"  {zone:10s}: {data['circumference_cm']} cm")
    print("  " + "-" * 40)

if __name__ == "__main__":
    main()