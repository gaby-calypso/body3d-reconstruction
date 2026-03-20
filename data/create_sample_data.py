"""
create_sample_data.py
---------------------
Generates a synthetic RGB image and depth map for testing the pipeline
without needing a real RealSense camera.

Run this once before running main.py:
    python data/create_sample_data.py
"""

from pathlib import Path
import numpy as np
import cv2

OUTPUT_DIR = Path("data/sample")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthetic RGB image ────────────────────────────────────────────────────────
H, W = 480, 640
rgb = np.full((H, W, 3), fill_value=[180, 160, 140], dtype=np.uint8)

# Head
cv2.ellipse(rgb, center=(320, 100), axes=(60, 70), angle=0,
            startAngle=0, endAngle=360, color=(100, 80, 60), thickness=-1)
# Torso
cv2.rectangle(rgb, pt1=(250, 165), pt2=(390, 380),
              color=(90, 70, 55), thickness=-1)

rgb_path = OUTPUT_DIR / "rgb.png"
cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print(f"  ✓ Saved: {rgb_path}")

# ── Synthetic depth map ────────────────────────────────────────────────────────
depth = np.full((H, W), fill_value=3000.0, dtype=np.float32)  # 3 m background
depth[80:390, 250:390] = 1500.0                                # person ~1.5 m away

# Realistic sensor noise
rng = np.random.default_rng(seed=42)
depth += rng.normal(loc=0, scale=5.0, size=depth.shape).astype(np.float32)

depth_path = OUTPUT_DIR / "depth.npy"
np.save(str(depth_path), depth)
print(f"  ✓ Saved: {depth_path}")

print("\nSample data ready. Now run: python main.py")