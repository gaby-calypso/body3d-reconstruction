"""
loader.py
---------
Responsible for loading input data into the pipeline.

Inputs:
    - An RGB image file (e.g., .png or .jpg)
    - A depth map stored as a NumPy array (.npy file)

Outputs:
    - rgb_image: np.ndarray of shape (H, W, 3), dtype uint8
    - depth_map: np.ndarray of shape (H, W),    dtype float32

Assumptions:
    - RGB and depth frames are spatially aligned (same resolution).
    - Depth values are in millimetres (standard for RealSense D455).
"""

from pathlib import Path
import numpy as np
import cv2


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """
    Load an RGB image from disk.

    OpenCV loads images as BGR by default. We convert to RGB immediately
    so every downstream module works in the standard RGB colour space.

    Args:
        image_path: Path to the image file (.png, .jpg, etc.)

    Returns:
        np.ndarray: RGB image with shape (H, W, 3) and dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If OpenCV cannot decode the file.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"RGB image not found: {image_path}")

    bgr = cv2.imread(str(image_path))

    if bgr is None:
        raise ValueError(f"Could not decode image file: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_depth_map(depth_path: str | Path) -> np.ndarray:
    """
    Load a depth map stored as a NumPy .npy file.

    Each value represents the distance in millimetres from the camera
    to that pixel.

    Args:
        depth_path: Path to the .npy depth file.

    Returns:
        np.ndarray: Depth map with shape (H, W), dtype float32.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the array is not 2D.
    """
    depth_path = Path(depth_path)

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    depth = np.load(str(depth_path))

    if depth.ndim != 2:
        raise ValueError(
            f"Expected a 2D depth array, got shape {depth.shape}. "
            "Make sure the .npy file contains a single depth frame."
        )

    return depth.astype(np.float32)


def load_frame(rgb_path: str | Path, depth_path: str | Path) -> dict:
    """
    Convenience function: load both RGB and depth in one call.

    This is the main function that main.py will use.
    Returns a dictionary so adding new fields later (e.g. camera
    intrinsics) does not break existing function signatures.

    Args:
        rgb_path:   Path to the RGB image file.
        depth_path: Path to the depth .npy file.

    Returns:
        dict with keys:
            'rgb'   -> np.ndarray (H, W, 3) uint8
            'depth' -> np.ndarray (H, W)    float32
    """
    rgb   = load_rgb_image(rgb_path)
    depth = load_depth_map(depth_path)

    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"Resolution mismatch: RGB is {rgb.shape[:2]}, "
            f"depth is {depth.shape}. They must match."
        )

    return {"rgb": rgb, "depth": depth}