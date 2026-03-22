"""
multi_view_loader.py
--------------------
Carga las 4 vistas (frontal, posterior, lateral izq., lateral der.)
desde archivos RGB + depth.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from typing import Optional

VIEW_CONFIG = {
    "frontal":     {"angle": 0.0,   "flip_x": False},
    "posterior":   {"angle": 180.0, "flip_x": True},
    "lateral_izq": {"angle": 90.0,  "flip_x": False},
    "lateral_der": {"angle": -90.0, "flip_x": True},
}


def load_view(
    rgb_path: str,
    depth_path: str,
    view_name: str,
    depth_scale: float = 1.0,
) -> dict:
    rgb_path   = Path(rgb_path)
    depth_path = Path(depth_path)

    rgb_bgr = cv2.imread(str(rgb_path))
    if rgb_bgr is None:
        raise FileNotFoundError(f"No se encontró la imagen RGB: {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    ext = depth_path.suffix.lower()
    if ext == ".npy":
        depth = np.load(str(depth_path)).astype("float32")
    elif ext in (".png", ".tiff", ".tif"):
        raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise FileNotFoundError(f"No se encontró el depth: {depth_path}")
        depth = raw.astype("float32")
    else:
        raise ValueError(f"Formato de depth no soportado: {ext}")

    depth *= depth_scale

    cfg = VIEW_CONFIG.get(view_name, {"angle": 0.0, "flip_x": False})

    valid = depth[depth > 0]
    print(f"  ✓ {view_name:15s}: rgb={rgb.shape}  depth={depth.shape}"
          f"  depth_range=[{valid.min():.0f}, {valid.max():.0f}] mm")

    return {
        "name":   view_name,
        "rgb":    rgb,
        "depth":  depth,
        "angle":  cfg["angle"],
        "flip_x": cfg["flip_x"],
    }


def load_all_views(
    data_dir: str,
    prefix_map: Optional[dict] = None,
    depth_ext: str = ".npy",
    depth_scale: float = 1.0,
) -> list:
    data_dir = Path(data_dir)

    if prefix_map is None:
        prefix_map = {
            "frontal":     ("frontal_rgb.png",     f"frontal_depth{depth_ext}"),
            "posterior":   ("posterior_rgb.png",   f"posterior_depth{depth_ext}"),
            "lateral_izq": ("lateral_izq_rgb.png", f"lateral_izq_depth{depth_ext}"),
            "lateral_der": ("lateral_der_rgb.png", f"lateral_der_depth{depth_ext}"),
        }

    views = []
    for view_name, (rgb_file, depth_file) in prefix_map.items():
        view = load_view(
            rgb_path    = data_dir / rgb_file,
            depth_path  = data_dir / depth_file,
            view_name   = view_name,
            depth_scale = depth_scale,
        )
        views.append(view)

    return views
