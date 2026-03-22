"""
preprocessing.py
----------------
Limpia y prepara el mapa de profundidad antes de la segmentación.

Pasos:
    1. Eliminar píxeles inválidos (0 y 65535) → los marca como NaN
    2. Recortar a un rango físicamente razonable (300–4000 mm)
    3. Rellenar huecos con inpainting (OpenCV)
    4. Suavizar ruido con filtro bilateral (preserva bordes)

Inputs:
    - depth: np.ndarray (H, W) float32 con valores en mm

Outputs:
    - depth limpio: np.ndarray (H, W) float32
"""

import numpy as np
import cv2


# Rango físico razonable para la RealSense D455 con una persona
DEPTH_MIN_MM = 300.0    # 30 cm — más cerca no es confiable
DEPTH_MAX_MM = 4000.0   # 4 m  — más lejos no nos interesa


def remove_invalid_pixels(depth: np.ndarray) -> np.ndarray:
    """
    Marca como NaN los píxeles inválidos del sensor.

    La RealSense D455 usa 0 para "sin datos" y 65535 para saturación.
    Ambos son inválidos y deben eliminarse antes de cualquier cálculo.

    Args:
        depth: mapa de profundidad (H, W) float32 en mm

    Returns:
        depth con píxeles inválidos reemplazados por NaN
    """
    cleaned = depth.copy()
    cleaned[(depth == 0)|(depth >= 65535)]     = np.nan
    # cleaned[depth >= 65535] = np.nan
    return cleaned


def clip_depth_range(depth: np.ndarray,
                     min_mm: float = DEPTH_MIN_MM,
                     max_mm: float = DEPTH_MAX_MM) -> np.ndarray:
    """
    Elimina valores fuera del rango físico razonable.

    Todo lo que esté más cerca de min_mm o más lejos de max_mm
    se descarta marcándolo como NaN.

    Args:
        depth:   mapa de profundidad (H, W) float32 en mm
        min_mm:  distancia mínima válida en mm
        max_mm:  distancia máxima válida en mm

    Returns:
        depth con valores fuera de rango reemplazados por NaN
    """
    clipped = depth.copy()
    clipped[depth < min_mm] = np.nan
    clipped[depth > max_mm] = np.nan
    return clipped


def fill_holes(depth: np.ndarray, max_hole_size: int = 5) -> np.ndarray:
    """
    Rellena huecos (NaN) usando inpainting de OpenCV.

    Inpainting estima el valor de cada píxel inválido basándose en
    sus vecinos válidos más cercanos. Es más preciso que un simple
    promedio porque respeta la continuidad de las superficies.

    Args:
        depth:         mapa de profundidad (H, W) float32 con NaN
        max_hole_size: radio en píxeles para el inpainting (default 5)

    Returns:
        depth con huecos rellenados, sin NaN
    """
    # OpenCV inpainting requiere uint16 — normalizamos a ese rango
    valid_mask = ~np.isnan(depth)

    if valid_mask.sum() == 0:
        raise ValueError("El mapa de profundidad no tiene píxeles válidos.")

    # Normalizar a 0–65534 para trabajar con uint16
    d_min = np.nanmin(depth)
    d_max = np.nanmax(depth)
    depth_norm = np.zeros_like(depth)
    depth_norm[valid_mask] = (
        (depth[valid_mask] - d_min) / (d_max - d_min) * 65534
    )
    depth_uint16 = depth_norm.astype(np.uint16)

    # La máscara de inpainting: 255 donde hay hueco, 0 donde hay dato
    inpaint_mask = (~valid_mask).astype(np.uint8) * 255

    # Aplicar inpainting
    filled_uint16 = cv2.inpaint(
        depth_uint16, inpaint_mask,
        inpaintRadius=max_hole_size,
        flags=cv2.INPAINT_TELEA
    )

    # Volver a mm
    filled = filled_uint16.astype(np.float32) / 65534 * (d_max - d_min) + d_min
    return filled


def smooth_depth(depth: np.ndarray,
                 diameter: int = 9,
                 sigma_color: float = 75.0,
                 sigma_space: float = 75.0) -> np.ndarray:
    """
    Suaviza el ruido del sensor con un filtro bilateral.

    A diferencia de un desenfoque gaussiano, el filtro bilateral
    preserva los bordes (p.ej. el contorno del cuerpo) mientras
    suaviza las superficies planas. Ideal para datos de profundidad.

    Args:
        depth:       mapa de profundidad (H, W) float32 sin NaN
        diameter:    tamaño del vecindario en píxeles (default 9)
        sigma_color: tolerancia de diferencia de profundidad (mm)
        sigma_space: tolerancia de distancia espacial (píxeles)

    Returns:
        depth suavizado (H, W) float32
    """
    # bilateralFilter requiere float32 normalizado a 0–1 o uint8
    # Usamos float32 directamente (OpenCV lo acepta)
    smoothed = cv2.bilateralFilter(
        depth, diameter, sigma_color, sigma_space
    )
    return smoothed


def preprocess_depth(depth: np.ndarray) -> np.ndarray:
    """
    Función principal: aplica todos los pasos en orden.

    Esta es la función que main.py llama. Ejecuta los 4 pasos
    en secuencia y reporta estadísticas en cada etapa.

    Args:
        depth: mapa de profundidad crudo (H, W) float32 en mm

    Returns:
        depth limpio y listo para segmentación (H, W) float32
    """
    total = depth.size

    # Paso 1 — eliminar inválidos
    depth = remove_invalid_pixels(depth)
    valid = np.sum(~np.isnan(depth))
    print(f"  [1] Inválidos eliminados: "
          f"{total - valid:,} píxeles → {valid:,} válidos "
          f"({100 * valid / total:.1f}%)")

    # Paso 2 — recortar rango
    depth = clip_depth_range(depth)
    valid = np.sum(~np.isnan(depth))
    print(f"  [2] Rango recortado ({DEPTH_MIN_MM:.0f}–{DEPTH_MAX_MM:.0f} mm): "
          f"{valid:,} válidos ({100 * valid / total:.1f}%)")

    # Paso 3 — rellenar huecos
    depth = fill_holes(depth)
    print(f"  [3] Huecos rellenados: "
          f"rango resultante {depth.min():.0f}–{depth.max():.0f} mm")

    # Paso 4 — suavizar ruido
    depth = smooth_depth(depth)
    print(f"  [4] Ruido suavizado: "
          f"profundidad promedio {depth.mean():.0f} mm")

    return depth