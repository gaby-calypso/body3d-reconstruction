"""
segmentation.py
---------------
Aísla la silueta del cuerpo humano combinando:
    1. ROI espacial (X, Y) — recorta la zona donde está la persona
    2. Rango de profundidad — separa persona del fondo por distancia
    3. Componente mayor    — elimina fragmentos residuales

Inputs:
    - depth_clean: np.ndarray (H, W) float32 en mm
    - rgb:         np.ndarray (H, W, 3) uint8

Outputs:
    - mask:       np.ndarray (H, W) uint8 — 255 persona, 0 fondo
    - depth_body: np.ndarray (H, W) float32
    - rgb_body:   np.ndarray (H, W, 3) uint8
"""

import numpy as np
import cv2


# ── Parámetros por defecto calibrados para esta sesión ────────────────────────
# Derivados de los valores de referencia del código Colab (escala 0-255 → mm)
# y del ROI (escala ~500px → 1280px)
DEFAULT_DEPTH_MIN_MM = 1000.0   # d_min=40  en escala 0-255 → mm
DEFAULT_DEPTH_MAX_MM = 1680.0   # d_max=75  en escala 0-255 → mm

DEFAULT_X_MIN = 450             # x_min=265 escalado a 1280px
DEFAULT_X_MAX = 750            # x_max=487 escalado a 1280px
DEFAULT_Y_MIN = 140               # sin restricción vertical superior
DEFAULT_Y_MAX = 555            # sin restricción vertical inferior


def create_roi_mask(shape: tuple,
                    x_min: int, x_max: int,
                    y_min: int, y_max: int) -> np.ndarray:
    """
    Crea una máscara de región de interés (ROI) espacial.

    Solo los píxeles dentro del rectángulo definido por
    (x_min, y_min) → (x_max, y_max) quedan habilitados.

    Args:
        shape: (H, W) de la imagen
        x_min, x_max: columnas izquierda y derecha
        y_min, y_max: filas superior e inferior

    Returns:
        máscara ROI (H, W) uint8 — 255 dentro, 0 fuera
    """
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255
    return mask


def create_depth_mask(depth: np.ndarray,
                      d_min_mm: float,
                      d_max_mm: float) -> np.ndarray:
    """
    Crea una máscara binaria por rango de profundidad.

    Solo los píxeles cuya profundidad está entre d_min_mm y
    d_max_mm quedan habilitados.

    Args:
        depth:    mapa de profundidad (H, W) float32 en mm
        d_min_mm: profundidad mínima en mm
        d_max_mm: profundidad máxima en mm

    Returns:
        máscara depth (H, W) uint8 — 255 en rango, 0 fuera
    """
    in_range = (depth >= d_min_mm) & (depth <= d_max_mm)
    return in_range.astype(np.uint8) * 255


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Limpia la máscara con operaciones morfológicas.

    - CLOSE grande: cierra huecos internos de la silueta
    - OPEN:         elimina artefactos pequeños externos
    - CLOSE final:  suaviza el contorno

    Args:
        mask: máscara binaria (H, W) uint8

    Returns:
        máscara refinada (H, W) uint8
    """
    # Cierre grande para tapar huecos internos (hueco blanco)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Apertura para eliminar artefactos pequeños externos
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Cierre suave final para alisar el contorno
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth)

    return mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Conserva solo el componente conectado más grande.

    Elimina cualquier fragmento residual que no sea la silueta
    principal de la persona.

    Args:
        mask: máscara binaria (H, W) uint8

    Returns:
        máscara con solo el componente más grande
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1

    clean_mask = np.zeros_like(mask)
    clean_mask[labels == largest_label] = 255
    return clean_mask


def apply_mask(rgb: np.ndarray,
               depth: np.ndarray,
               mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica la máscara al RGB y al depth.

    Args:
        rgb:   imagen RGB (H, W, 3) uint8
        depth: mapa de profundidad (H, W) float32
        mask:  máscara binaria (H, W) uint8

    Returns:
        rgb_body:   RGB con fondo negro
        depth_body: depth solo de la persona
    """
    binary = (mask == 255)

    rgb_body = np.zeros_like(rgb)
    rgb_body[binary] = rgb[binary]

    depth_body = np.zeros_like(depth)
    depth_body[binary] = depth[binary]

    return rgb_body, depth_body


def segment_body(rgb: np.ndarray,
                 depth: np.ndarray,
                 d_min_mm: float = DEFAULT_DEPTH_MIN_MM,
                 d_max_mm: float = DEFAULT_DEPTH_MAX_MM,
                 x_min: int = DEFAULT_X_MIN,
                 x_max: int = DEFAULT_X_MAX,
                 y_min: int = DEFAULT_Y_MIN,
                 y_max: int = DEFAULT_Y_MAX) -> dict:
    """
    Función principal: segmentación completa por ROI + profundidad.

    Replica la estrategia del código Colab de referencia:
    ROI espacial + rango de profundidad + componente mayor.

    Args:
        rgb:      imagen RGB (H, W, 3) uint8
        depth:    depth preprocesado (H, W) float32 en mm
        d_min_mm: profundidad mínima de la persona en mm
        d_max_mm: profundidad máxima de la persona en mm
        x_min:    columna izquierda del ROI
        x_max:    columna derecha del ROI
        y_min:    fila superior del ROI
        y_max:    fila inferior del ROI

    Returns:
        dict con claves:
            'mask'        -> np.ndarray (H, W) uint8
            'rgb_body'    -> np.ndarray (H, W, 3) uint8
            'depth_body'  -> np.ndarray (H, W) float32
            'body_pixels' -> int
    """
    H, W = depth.shape

    # Paso 1 — ROI espacial
    roi_mask = create_roi_mask((H, W), x_min, x_max, y_min, y_max)
    print(f"  [1] ROI: x={x_min}-{x_max}, y={y_min}-{y_max}")

    # Paso 2 — rango de profundidad
    depth_mask = create_depth_mask(depth, d_min_mm, d_max_mm)
    print(f"  [2] Rango profundidad: {d_min_mm:.0f}-{d_max_mm:.0f} mm")

    # Paso 3 — combinar ROI + profundidad
    combined = cv2.bitwise_and(roi_mask, depth_mask)

    # Paso 4 — morfología
    mask = refine_mask(combined)

    # Paso 5 — componente mayor
    mask = keep_largest_component(mask)
    body_pixels = int(np.sum(mask == 255))
    print(f"  [3] Silueta principal: {body_pixels:,} píxeles "
          f"({100 * body_pixels / mask.size:.1f}% de la imagen)")

    # Paso 6 — aplicar máscara
    rgb_body, depth_body = apply_mask(rgb, depth, mask)
    valid = depth_body[depth_body > 0]
    if len(valid) > 0:
        print(f"  [4] Depth silueta: {valid.min():.0f}–{valid.max():.0f} mm")

    return {
        "mask":        mask,
        "rgb_body":    rgb_body,
        "depth_body":  depth_body,
        "body_pixels": body_pixels,
    }