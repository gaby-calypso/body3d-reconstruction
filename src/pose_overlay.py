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
from pathlib import Path
from typing import Optional

# El modelo puede estar en cualquiera de estas rutas — antes solo se
# revisaba /tmp/, pero el mensaje de la GUI hablaba de "models/..." (una
# carpeta distinta dentro del propio repo) y esa ruta nunca se comprobaba
# de verdad, así que el aviso "no encontrado" salía aunque el modelo
# estuviera ahí. Ahora se busca en ambas, y si no aparece en ninguna se
# intenta descargar automáticamente a /tmp/.
_REPO_ROOT = Path(__file__).resolve().parents[1]   # src/ -> raíz del repo
MODEL_CANDIDATES = [
    _REPO_ROOT / "models" / "pose_landmarker_full.task",
    Path("/tmp/pose_landmarker_full.task"),
]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

# Mantenido por compatibilidad con el resto del módulo/otros imports —
# apunta a la ruta de descarga por defecto si ninguna de las candidatas
# existe todavía.
MODEL_PATH = str(MODEL_CANDIDATES[1])


def _find_model_path() -> Optional[str]:
    for p in MODEL_CANDIDATES:
        if p.exists() and p.stat().st_size > 0:
            return str(p)
    return None


def _try_download_model() -> Optional[str]:
    """Intenta descargar el modelo a /tmp/ si no se encontró en disco."""
    dest = MODEL_CANDIDATES[1]
    try:
        import urllib.request
        print(f"  Modelo de landmarks no encontrado — descargando a {dest} ...")
        urllib.request.urlretrieve(MODEL_URL, str(dest))
        if dest.exists() and dest.stat().st_size > 0:
            print("  ✓ Modelo descargado correctamente.")
            return str(dest)
        print("  ⚠ La descarga terminó pero el archivo quedó vacío.")
    except Exception as e:
        print(f"  ⚠ No se pudo descargar el modelo automáticamente: {e}")
    return None


def model_available() -> bool:
    """True si el modelo de landmarks (.task) está disponible, descargándolo
    automáticamente a /tmp/ si hace falta y hay conexión a internet."""
    return _find_model_path() is not None or _try_download_model() is not None

# ── Offset de paralaje RGB → Depth (calibrado con calibrate_parallax.py) ─────
# El sensor RGB y el sensor de profundidad están físicamente separados en la
# D455, por lo que los landmarks detectados en RGB deben desplazarse antes
# de dibujarse sobre el depth.
# Offset para trasladar landmarks del RGB al espacio del depth.
# Valor = opuesto al paralaje medido en calibrate_parallax.py
# (calibrate mueve depth sobre RGB; aqui movemos landmarks de RGB a depth)
PARALLAX_X = 0    # offset RGB→Depth adicional, en px (ver nota abajo)
PARALLAX_Y = 0

# NOTA IMPORTANTE: camera.py ya alinea depth↔color por hardware con
# rs.align(rs.stream.color) antes de guardar cualquier captura — eso
# significa que el depth YA viene reproyectado al sistema de coordenadas
# del RGB, sin paralaje. El valor histórico de este offset (27, -17) se
# midió con calibrate_parallax.py sobre streams SIN alinear; aplicarlo
# encima de capturas ya alineadas introduce un corrimiento artificial
# (justamente el "landmarks corridos a la derecha" que se veía).
# Si en tu setup particular sigue quedando un pequeño desfase residual,
# vuelve a correr calibrate_parallax.py sobre una captura real (ya
# alineada) y ajusta estos dos valores — deberían ser chicos (unos pocos
# píxeles), no del orden de 20-30px.

# Resolución con la que se calibró el offset de arriba. Si la imagen real
# (RGB/depth) tiene otro tamaño (p.ej. 640x480, resolución típica de la
# D455 configurada en camera.py), aplicar 27/-17 píxeles TAL CUAL queda
# desproporcionado — es un offset mucho más grande en relación al tamaño
# de la imagen, y los landmarks aparecen "corridos" respecto al cuerpo.
# _scaled_parallax() reescala el offset proporcionalmente al tamaño real.
_PARALLAX_REF_W, _PARALLAX_REF_H = 1280, 720


def _scaled_parallax(w: int, h: int) -> tuple[int, int]:
    """Devuelve (offset_x, offset_y) reescalados al tamaño real (w, h)."""
    px = int(round(PARALLAX_X * w / _PARALLAX_REF_W))
    py = int(round(PARALLAX_Y * h / _PARALLAX_REF_H))
    return px, py

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

        model_path = _find_model_path() or _try_download_model()
        if not model_path:
            print("  ⚠ Modelo no encontrado en ninguna de estas rutas:")
            for p in MODEL_CANDIDATES:
                print(f"    - {p}")
            print("  Y no se pudo descargar automáticamente. Descárgalo con:")
            print('  curl -k -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -o /tmp/pose_landmarker_full.task')
            return None

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
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

        # MediaPipe siempre devuelve los 33 puntos del esqueleto, pero
        # algunos pueden venir con visibilidad muy baja (persona parcialmente
        # fuera de cuadro, oculta, etc). No se descartan aquí — se dejan
        # pasar tal cual y es _draw_on_frame quien decide, punto por punto,
        # cuáles dibujar según su visibilidad, así que nunca hace falta que
        # "todos" los puntos se detecten bien para poder graficar el resto.
        landmarks = result.pose_landmarks[0]
        n_visibles = sum(1 for lm in landmarks
                          if float(getattr(lm, "visibility", 1.0)) >= 0.3)
        print(f"  ✓ Persona detectada — {len(landmarks)} landmarks "
              f"({n_visibles} con buena visibilidad)")
        return landmarks

    except Exception as e:
        print(f"  ⚠ Error en MediaPipe: {e}")
        return None


def _landmarks_to_pixels(landmarks: list, H: int, W: int) -> list:
    """Convierte landmarks normalizados (0-1) a píxeles.

    Cada landmark se procesa de forma independiente: si uno viene con
    datos raros (por baja confianza de detección), se descarta ese punto
    puntual en vez de abortar la conversión completa.
    """
    points = []
    for lm in landmarks:
        try:
            x = max(0, min(W - 1, int(lm.x * W)))
            y = max(0, min(H - 1, int(lm.y * H)))
            vis = float(getattr(lm, "visibility", 1.0))
        except Exception:
            x, y, vis = 0, 0, 0.0   # visibilidad 0 → _draw_on_frame lo ignora
        points.append((x, y, vis))
    return points


def _draw_on_frame(img_bgr: np.ndarray, points: list) -> np.ndarray:
    """Dibuja conexiones y landmarks sobre imagen BGR.

    Solo se dibujan los puntos/conexiones cuya visibilidad es aceptable
    (>= 0.3) — los que MediaPipe no logró ubicar con confianza simplemente
    no se pintan, en vez de intentar graficar una posición inventada o
    fallar. Cada punto/conexión se dibuja dentro de su propio try/except
    para que un dato puntual corrupto no tumbe el resto del dibujo.
    """
    overlay = img_bgr.copy()

    # Conexiones
    for i, j in POSE_CONNECTIONS:
        try:
            if i >= len(points) or j >= len(points):
                continue
            x1, y1, v1 = points[i]
            x2, y2, v2 = points[j]
            if v1 < 0.3 or v2 < 0.3:
                continue
            cv2.line(overlay, (x1, y1), (x2, y2), COLOR_SKELETON, 1,
                     lineType=cv2.LINE_AA)
        except Exception:
            continue

    # Puntos
    for idx, (x, y, vis) in enumerate(points):
        try:
            if vis < 0.3:
                continue
            color = _landmark_color(idx)
            cv2.circle(overlay, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            # Número solo en landmarks clave
            if idx in (0, 11, 12, 23, 24, 25, 26, 27, 28):
                cv2.putText(overlay, str(idx), (x + 6, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        except Exception:
            continue

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
    1. Detecta landmarks en la imagen RGB
    2. Aplica offset de paralaje a las coordenadas
    3. Dibuja SOLO sobre la imagen de profundidad

    Args:
        rgb_bgr: imagen BGR (H, W, 3) — formato cv2.imread
        depth:   mapa de profundidad (H, W) float32 en mm

    Returns:
        (rgb_bgr_original, depth_annotated, success)
    """
    try:
        H, W = rgb_bgr.shape[:2]

        # Paso 1: detectar en RGB (convertir BGR→RGB para MediaPipe)
        rgb_for_mp = cv2.cvtColor(rgb_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        landmarks = _detect_landmarks(rgb_for_mp)
        if landmarks is None:
            return rgb_bgr, _depth_to_bgr(depth), False

        # Paso 2: convertir landmarks a píxeles en espacio RGB
        points_rgb = _landmarks_to_pixels(landmarks, H, W)

        # Paso 3: aplicar offset de paralaje (reescalado al tamaño real de esta
        # imagen) → coordenadas en espacio depth
        off_x, off_y = _scaled_parallax(W, H)
        points_depth = []
        for (x, y, vis) in points_rgb:
            x_d = max(0, min(W - 1, x + off_x))
            y_d = max(0, min(H - 1, y + off_y))
            points_depth.append((x_d, y_d, vis))

        # Paso 4: dibujar SOLO sobre la imagen de profundidad (cada punto
        # se dibuja según su propia visibilidad — ver _draw_on_frame)
        depth_annotated = _draw_on_frame(_depth_to_bgr(depth), points_depth)

        return rgb_bgr, depth_annotated, True

    except Exception as e:
        # Última red de seguridad: cualquier error inesperado (imagen
        # corrupta, forma inválida, etc.) se reporta como detección
        # fallida en vez de dejar que la excepción suba hasta el slot de
        # Qt que llamó a esta función — un ImportError similar fue lo que
        # antes cerraba la aplicación entera sin avisar.
        print(f"  ⚠ Error dibujando landmarks: {e}")
        try:
            return rgb_bgr, _depth_to_bgr(depth), False
        except Exception:
            return rgb_bgr, depth, False