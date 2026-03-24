"""
pose_overlay.py
---------------
Detecta los 33 landmarks de MediaPipe Pose y los dibuja sobre
la imagen RGB y el mapa de profundidad.

Requiere MediaPipe >= 0.10 con la nueva API tasks.
El modelo pose_landmarker_full.task debe estar en /tmp/ — descargarlo con:
    curl -k -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -o /tmp/pose_landmarker_full.task

Uso:
    from src.pose_overlay import draw_landmarks_on_images
    rgb_annotated, depth_annotated = draw_landmarks_on_images(rgb_bgr, depth)

Inputs:
    rgb_bgr: np.ndarray (H, W, 3) uint8  — imagen en BGR (formato cv2.imread)
    depth:   np.ndarray (H, W)   float32 — mapa de profundidad en mm

Outputs:
    rgb_annotated:   np.ndarray (H, W, 3) BGR — RGB con landmarks
    depth_annotated: np.ndarray (H, W, 3) BGR — depth coloreado con landmarks
    success:         bool
"""

import numpy as np
import cv2
import os
from typing import Optional

MODEL_PATH = "/tmp/pose_landmarker_full.task"

# ── Offset de paralaje RGB → Depth (calibrado con calibrate_parallax.py) ─────
# El sensor RGB y el sensor de profundidad están físicamente separados en la
# D455, por lo que los landmarks detectados en RGB deben desplazarse antes
# de dibujarse sobre el depth.
# Offset para trasladar landmarks del RGB al espacio del depth.
# Valor = opuesto al paralaje medido en calibrate_parallax.py
# (calibrate mueve depth sobre RGB; aqui movemos landmarks de RGB a depth)
PARALLAX_X = 27    # opuesto de -27
PARALLAX_Y = -17   # opuesto de +17

# ── Colores BGR por grupo de landmarks ───────────────────────────────────────
COLOR_FACE      = (180, 180, 180)
COLOR_SHOULDERS = (219, 152,  52)
COLOR_ARMS      = (113, 204,  46)
COLOR_HANDS     = ( 15, 196, 241)
COLOR_HIPS      = (182,  89, 155)
COLOR_LEGS      = ( 60,  76, 231)
COLOR_FEET      = ( 34, 126, 230)
COLOR_SKELETON  = (200, 200, 200)

# ── Conexiones del esqueleto ──────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
]

def _landmark_color(idx: int) -> tuple:
    if idx <= 10:             return COLOR_FACE
    if idx in (11, 12):       return COLOR_SHOULDERS
    if idx in (13,14,15,16):  return COLOR_ARMS
    if 17 <= idx <= 22:       return COLOR_HANDS
    if idx in (23, 24):       return COLOR_HIPS
    if idx in (25,26,27,28):  return COLOR_LEGS
    return COLOR_FEET


def _detect_landmarks(rgb_image: np.ndarray) -> Optional[list]:
    """
    Detecta landmarks usando MediaPipe Tasks API (>= 0.10).

    Args:
        rgb_image: imagen en formato RGB uint8

    Returns:
        lista de 33 landmarks normalizados, o None si falla
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not os.path.exists(MODEL_PATH):
            print(f"  ⚠ Modelo no encontrado en {MODEL_PATH}")
            print("  Descárgalo con:")
            print('  curl -k -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -o /tmp/pose_landmarker_full.task')
            return None

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        with mp_vision.PoseLandmarker.create_from_options(options) as detector:
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image.astype(np.uint8),
            )
            result = detector.detect(mp_img)

        if not result.pose_landmarks:
            print("  ⚠ No se detectó ninguna persona en la imagen")
            return None

        print(f"  ✓ Persona detectada — {len(result.pose_landmarks[0])} landmarks")
        return result.pose_landmarks[0]

    except Exception as e:
        print(f"  ⚠ Error en MediaPipe: {e}")
        return None


def _landmarks_to_pixels(landmarks: list, H: int, W: int) -> list:
    """Convierte landmarks normalizados (0-1) a píxeles."""
    points = []
    for lm in landmarks:
        x = max(0, min(W - 1, int(lm.x * W)))
        y = max(0, min(H - 1, int(lm.y * H)))
        vis = float(getattr(lm, "visibility", 1.0))
        points.append((x, y, vis))
    return points


def _draw_on_frame(img_bgr: np.ndarray, points: list) -> np.ndarray:
    """Dibuja conexiones y landmarks sobre imagen BGR."""
    overlay = img_bgr.copy()

    # Conexiones
    for i, j in POSE_CONNECTIONS:
        if i >= len(points) or j >= len(points):
            continue
        x1, y1, v1 = points[i]
        x2, y2, v2 = points[j]
        if v1 < 0.3 or v2 < 0.3:
            continue
        cv2.line(overlay, (x1, y1), (x2, y2), COLOR_SKELETON, 1,
                 lineType=cv2.LINE_AA)

    # Puntos
    for idx, (x, y, vis) in enumerate(points):
        if vis < 0.3:
            continue
        color = _landmark_color(idx)
        cv2.circle(overlay, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        # Número solo en landmarks clave
        if idx in (0, 11, 12, 23, 24, 25, 26, 27, 28):
            cv2.putText(overlay, str(idx), (x + 6, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.9, img_bgr, 0.1, 0)


def _depth_to_bgr(depth: np.ndarray) -> np.ndarray:
    """Convierte mapa de profundidad a imagen BGR coloreada."""
    valid = depth[(depth > 300) & (depth < 4000)]
    d_norm = np.zeros_like(depth, dtype=np.uint8)
    if len(valid) > 0:
        mask = (depth > 300) & (depth < 4000)
        d_min, d_max = valid.min(), valid.max()
        if d_max > d_min:
            d_norm[mask] = ((depth[mask] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)


def draw_landmarks_on_images(
    rgb_bgr: np.ndarray,
    depth: np.ndarray,
) -> tuple:
    """
    Detecta los 33 landmarks y los dibuja sobre RGB y depth.

    Args:
        rgb_bgr: imagen BGR (H, W, 3) uint8 — formato cv2.imread / STATE.rgbs
        depth:   mapa de profundidad (H, W) float32 en mm

    Returns:
        (rgb_annotated, depth_annotated, success)
        rgb_annotated:   BGR con landmarks dibujados
        depth_annotated: depth coloreado con landmarks dibujados
        success:         True si se detectaron landmarks
    """
    H, W = rgb_bgr.shape[:2]

    # STATE.rgbs guarda BGR — convertir a RGB para MediaPipe
    rgb_for_mp = cv2.cvtColor(rgb_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)

    landmarks = _detect_landmarks(rgb_for_mp)
    if landmarks is None:
        return None, None, False

    points = _landmarks_to_pixels(landmarks, H, W)

    # ── Aplicar offset de paralaje para el depth ─────────────────────────────
    # Los landmarks se detectaron sobre el RGB. Para dibujarlos sobre el depth
    # hay que desplazarlos por el paralaje calibrado entre ambas cámaras.
    H, W = rgb_bgr.shape[:2]
    points_depth = []
    for (x, y, vis) in points:
        x_d = max(0, min(W - 1, x + PARALLAX_X))
        y_d = max(0, min(H - 1, y + PARALLAX_Y))
        points_depth.append((x_d, y_d, vis))

    # Dibujar sobre BGR original (RGB sin offset)
    rgb_annotated   = _draw_on_frame(rgb_bgr.copy(), points)
    # Dibujar sobre depth con offset de paralaje aplicado
    depth_annotated = _draw_on_frame(_depth_to_bgr(depth), points_depth)

    return rgb_annotated, depth_annotated, True