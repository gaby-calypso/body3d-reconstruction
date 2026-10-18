"""
multi_view_pipeline.py
-----------------------
Pipeline multi-vista usando el mismo código que funciona para vista única.
Aplica segment_body + reconstruct_pointcloud + measurements a cada vista
por separado, luego combina los resultados para el reporte.

Automatización con MediaPipe:
    Las zonas de medición se detectan automáticamente usando los landmarks
    de MediaPipe Pose en cada vista. El marcado manual en VIEW_ZONES sigue
    disponible como override — si una zona está marcada manualmente, tiene
    prioridad sobre la detección automática.
"""

from __future__ import annotations
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from typing import Optional

from src.preprocessing   import preprocess_depth
from src.segmentation    import segment_body
from src.reconstruction  import reconstruct_pointcloud
from src.measurements    import (measure_row_range, compute_circumference,
                                  pixels_to_mm_width, build_zones)

# ── Zonas marcadas manualmente desde la GUI (override) ────────────────────────
# Si están vacías, se usan las zonas automáticas de MediaPipe.
VIEW_ZONES = {
    "frontal":     {},
    "posterior":   {},
    "lateral_izq": {},
    "lateral_der": {},
}

# Zonas esperadas por vista
# Brazo, muslo y rodilla se miden POR LADO (izq/der).
#   · frontal / posterior: se ve el ancho de cada lado → ambos lados.
#   · lateral_izq / lateral_der: se ve el grosor (profundidad) del lado que
#     da a la cámara → solo el lado correspondiente a esa vista.
ZONES_PER_VIEW = {
    "frontal":     ["cuello", "pecho", "brazo_izq", "brazo_der", "cintura",
                    "cadera", "muslo_izq", "muslo_der",
                    "rodilla_izq", "rodilla_der"],
    "posterior":   ["brazo_izq", "brazo_der", "cintura", "cadera",
                    "muslo_izq", "muslo_der", "rodilla_izq", "rodilla_der"],
    "lateral_izq": ["brazo_izq", "muslo_izq", "rodilla_izq",
                    "prof_pecho", "prof_cintura"],
    "lateral_der": ["brazo_der", "muslo_der", "rodilla_der",
                    "prof_pecho", "prof_cintura"],
}

# Relación profundidad/ancho (anatómica) para el tronco y el cuello. Se usa
# cuando no se logra medir el grosor en la vista lateral. SIN esto, la
# profundidad caía al "delta" frontal (p90-p10 de la profundidad dentro de
# la fila, solo 4-9 cm: mide cuánto se curva el frente, no el grosor real
# del cuerpo) y las circunferencias salían entre 10 y 30 % cortas.
DEPTH_WIDTH_RATIO = {
    "cuello":  0.85,   # el cuello es casi circular (siempre se usa)
    "pecho":   0.70,
    "cintura": 0.65,
    "cadera":  0.65,
}

# Relación profundidad/ancho para brazo, muslo y rodilla cuando NO hay vista
# lateral de ese lado (las extremidades son casi circulares).
LIMB_DEPTH_RATIO = 0.90


def _base_zone(zone: str) -> str:
    """'muslo_izq' → 'muslo'; 'cintura' → 'cintura'."""
    for suf in ("_izq", "_der"):
        if zone.endswith(suf):
            return zone[: -len(suf)]
    return zone

IMG_CENTER_X = 320   # centro horizontal de una imagen de 640x480
BAND         = 8   # ±px alrededor del landmark para promediar filas

# Offset de paralaje RGB → Depth (calibrado con calibrate_parallax.py)
# Los landmarks se detectan en RGB — hay que desplazarlos para que
# coincidan con las posiciones correctas en el depth.
# Calibrado sobre imágenes de referencia de 1280x720 (ver
# calibrate_parallax.py) — si la imagen real tiene otro tamaño (p.ej.
# 640x480, resolución típica de la D455 en camera.py), este offset fijo
# queda desproporcionado y los landmarks terminan "corridos" respecto al
# cuerpo. _lm_x/_lm_y reescalan el offset al tamaño real de cada imagen.
PARALLAX_X = 0    # offset RGB→Depth adicional (ver nota en pose_overlay.py:
PARALLAX_Y = 0    # camera.py ya alinea depth↔color con rs.align, así que
                   # este offset manual debería ser 0 salvo recalibración)
_PARALLAX_REF_W, _PARALLAX_REF_H = 1280, 720


# ── Configuración por vista ────────────────────────────────────────────────────
VIEW_CONFIG = {
    "frontal": {
        "d_min": 1410, "d_max": 1790,
        "x_min": 198,  "x_max": 378,
        "y_min": 0,    "y_max": 417,
    },
    "posterior": {
        "d_min": 1180, "d_max": 1720,
        "x_min": 205,  "x_max": 390,
        "y_min": 0,    "y_max": 417,
    },
    "lateral_izq": {
        "d_min": 1090, "d_max": 1490,
        "x_min": 224,  "x_max": 358,
        "y_min": 0,    "y_max": 446,
    },
    "lateral_der": {
        "d_min": 930,  "d_max": 1750,
        "x_min": 211,  "x_max": 365,
        "y_min": 0,    "y_max": 417,
    },
}

# Fracción del ancho a usar por zona (para excluir brazos)
ZONE_WIDTH_PCT = {
    "cuello":  100.0,
    "pecho":    70.0,
    "brazo":   100.0,
    "cintura": 100.0,
    "cadera":   80.0,
    "muslo":   100.0,
    "rodilla": 100.0,
}

# ── Posiciones anatómicas normalizadas (fallback sin MediaPipe) ───────────────
ANATOMICAL_NORM = {
    "cuello":  0.10,
    "pecho":   0.28,
    "brazo":   0.32,
    "cintura": 0.48,
    "cadera":  0.58,
    "muslo":   0.72,
    "rodilla": 0.88,
}


# ── Detección de landmarks por vista ──────────────────────────────────────────

def _detect_landmarks_for_view(rgb: np.ndarray) -> Optional[list]:
    """
    Detecta los 33 landmarks de MediaPipe en una imagen BGR.
    Convierte BGR→RGB aquí antes de llamar a _detect_landmarks.
    """
    try:
        from src.pose_overlay import _detect_landmarks
        import cv2 as _cv2
        # _detect_landmarks espera RGB — convertir desde BGR (formato cv2/STATE.rgbs)
        rgb_for_mp = _cv2.cvtColor(rgb.astype(np.uint8), _cv2.COLOR_BGR2RGB)
        return _detect_landmarks(rgb_for_mp)
    except Exception as e:
        print(f"  ⚠ MediaPipe falló: {e}")
        return None


def _lm_y(landmarks: list, idx: int, H: int,
          apply_parallax: bool = False) -> int:
    """Convierte landmark normalizado a coordenada Y en píxeles.
    Si apply_parallax=True aplica el offset RGB→Depth."""
    y = int(landmarks[idx].y * H)
    if apply_parallax:
        off_y = int(round(PARALLAX_Y * H / _PARALLAX_REF_H))
        y = max(0, min(H-1, y + off_y))
    return y


def _lm_x(landmarks: list, idx: int, W: int,
           apply_parallax: bool = False) -> int:
    """Convierte landmark normalizado a coordenada X en píxeles.
    Si apply_parallax=True aplica el offset RGB→Depth."""
    x = int(landmarks[idx].x * W)
    if apply_parallax:
        off_x = int(round(PARALLAX_X * W / _PARALLAX_REF_W))
        x = max(0, min(W-1, x + off_x))
    return x


def _band(y_center: int, H: int, band: int = BAND) -> tuple[int, int]:
    """Devuelve (y_start, y_end) alrededor de y_center, clampeado a [0, H-1]."""
    return max(0, y_center - band), min(H - 1, y_center + band)


def _silhouette_x_bounds(mask: np.ndarray, y_center: int,
                           band: int = BAND) -> tuple[int, int]:
    """
    Obtiene los límites reales X de la silueta en una banda Y.

    En lugar de usar estimaciones geométricas, lee directamente los
    píxeles activos de la máscara en la banda y_center ± band.
    Esto garantiza que el cuadrante siempre ciñe el contorno real
    del cuerpo, independientemente de la postura.

    Returns:
        (x1, x2) — columnas mínima y máxima con píxeles activos
    """
    y1 = max(0, y_center - band)
    y2 = min(mask.shape[0] - 1, y_center + band)
    band_slice = mask[y1:y2+1, :]
    cols = np.where(band_slice.max(axis=0) == 255)[0]
    if len(cols) < 2:
        return 0, mask.shape[1]
    return int(cols.min()), int(cols.max())


def _run_bounds(mask, y_center, x_center, x_lo, x_hi, band=BAND):
    """
    Límites X del tramo continuo de silueta que contiene (o queda más
    cerca de) x_center, dentro de la banda y_center±band y de [x_lo, x_hi].
    Sirve para aislar UNA pierna aunque la otra esté pegada.
    Devuelve None si no hay silueta en esa banda.
    """
    if mask is None:
        return None
    H, W = mask.shape[:2]
    y1 = max(0, y_center - band)
    y2 = min(H - 1, y_center + band)
    x_lo = max(0, x_lo); x_hi = min(W, x_hi)
    if x_hi - x_lo < 3:
        return None
    sl  = mask[y1:y2 + 1, x_lo:x_hi] == 255
    occ = sl.mean(axis=0) >= 0.5
    if not occ.any():
        return None
    xs = np.where(occ)[0]
    # separar en tramos continuos
    runs, start = [], xs[0]
    for a, b in zip(xs[:-1], xs[1:]):
        if b != a + 1:
            runs.append((start, a)); start = b
    runs.append((start, xs[-1]))
    xc = x_center - x_lo
    best = min(runs, key=lambda r: 0 if r[0] <= xc <= r[1]
               else min(abs(xc - r[0]), abs(xc - r[1])))
    return int(best[0] + x_lo), int(best[1] + x_lo + 1)


def _auto_zones_frontal_posterior(
    landmarks: list, H: int, W: int,
    mask: np.ndarray = None,
) -> dict:
    """
    Calcula zonas automáticas para vistas frontales y posteriores.

    Estrategia combinada:
    - MediaPipe da la posición Y de cada zona (dónde está anatómicamente)
    - La silueta de la máscara da los límites X reales (cuánto mide)
    - Para cada zona se excluyen los brazos usando los landmarks como guía

    Zonas calculadas:
        cuello    → entre landmarks de nariz(0) y hombros(11,12)
                    X acotado al cilindro del cuello (excluyendo hombros)
        pecho     → 25% del torso desde hombros
                    X desde el contorno real de la silueta
        cintura   → 50% del torso (zona más estrecha)
                    X desde el contorno real, excluyendo brazos
        cadera    → altura de landmarks 23/24
                    X desde el contorno real de la silueta
        brazo_izq → landmark codo 13, X alrededor del brazo
        brazo_der → landmark codo 14, X alrededor del brazo

    Args:
        mask: máscara segmentada opcional — si se provee, los límites
              X vienen del contorno real en lugar de estimaciones

    Returns:
        dict {zona: {"y_center", "rows", "width_pct", "x1", "x2"}}
    """
    # Aplicar paralaje: landmarks detectados en RGB,
    # pero las zonas se miden en depth
    p = True  # apply_parallax

    y_sh  = (_lm_y(landmarks, 11, H, p) + _lm_y(landmarks, 12, H, p)) // 2
    y_hi  = (_lm_y(landmarks, 23, H, p) + _lm_y(landmarks, 24, H, p)) // 2
    torso = y_hi - y_sh

    if torso < 10:
        return {}

    # ── Posiciones Y anatómicas desde landmarks (con paralaje aplicado) ───────
    y_nariz   = _lm_y(landmarks, 0, H, p)
    y_cuello  = y_nariz + int((y_sh - y_nariz) * 0.75)

    y_pecho   = y_sh + int(0.25 * torso)
    y_cintura = y_sh + int(0.55 * torso)
    y_cadera  = y_hi - int(0.08 * torso)

    # ── X de hombros y codos (con paralaje) ──────────────────────────────────
    x_sh_izq  = _lm_x(landmarks, 11, W, p)
    x_sh_der  = _lm_x(landmarks, 12, W, p)
    x_sh_min  = min(x_sh_der, x_sh_izq)
    x_sh_max  = max(x_sh_der, x_sh_izq)

    x_co_izq  = _lm_x(landmarks, 13, W, p)
    x_co_der  = _lm_x(landmarks, 14, W, p)
    y_co_izq  = _lm_y(landmarks, 13, H, p)
    y_co_der  = _lm_y(landmarks, 14, H, p)

    # Brazos: punto medio entre hombro y codo
    y_brazo_izq = (y_sh + y_co_izq) // 2
    y_brazo_der = (y_sh + y_co_der) // 2
    x_brazo_izq = (x_sh_izq + x_co_izq) // 2
    x_brazo_der = (x_sh_der + x_co_der) // 2
    # Semiancho de la ventana del brazo: calibrado en 1280 px de ancho
    brazo_half  = max(6, int(round(22 * W / 1280.0)))
    brazo_vhalf = max(10, int(round(34 * H / 720.0)))   # semialto (brazo horizontal)

    def _arm_zone(sh_idx, el_idx):
        """Zona del brazo: vertical (colgando) u horizontal (extendido)."""
        xs_, ys_ = _lm_x(landmarks, sh_idx, W, p), _lm_y(landmarks, sh_idx, H, p)
        xe_, ye_ = _lm_x(landmarks, el_idx, W, p), _lm_y(landmarks, el_idx, H, p)
        xm, ym = (xs_ + xe_) // 2, (ys_ + ye_) // 2
        if abs(xe_ - xs_) > 1.2 * abs(ye_ - ys_):          # extendido
            return {
                "y_center":  ym,
                "rows":      (max(0, ym - brazo_vhalf), min(H - 1, ym + brazo_vhalf)),
                "width_pct": 100.0,
                "x1": max(0, xm - brazo_half), "x2": min(W, xm + brazo_half),
                "orient": "h",
            }
        return {                                            # colgando
            "y_center":  ym,
            "rows":      _band(ym, H),
            "width_pct": 100.0,
            "x1": max(0, xm - brazo_half), "x2": min(W, xm + brazo_half),
            "orient": "v",
        }

    def zone_from_mask(y_c, zone_name):
        """
        Construye cfg de zona usando límites reales de la silueta.
        MediaPipe da la Y, la máscara da el X real del contorno.
        Cada zona tiene su lógica anatómica para excluir extremidades.
        """
        if mask is not None:
            x1_raw, x2_raw = _silhouette_x_bounds(mask, y_c)
        else:
            x1_raw, x2_raw = 0, W

        if zone_name == "cuello":
            # Cuello: más estrecho que hombros — 20% interno del ancho entre hombros
            margin = int((x_sh_max - x_sh_min) * 0.20)
            x1 = max(x1_raw, x_sh_min + margin)
            x2 = min(x2_raw, x_sh_max - margin)

        elif zone_name == "pecho":
            # Pecho: ancho de hombro a hombro con pequeño margen
            x1 = max(x1_raw, x_sh_min - 10)
            x2 = min(x2_raw, x_sh_max + 10)

        elif zone_name == "cintura":
            # Cintura: centrada en el torso usando los hombros como referencia
            # Tomar el centro entre hombros y expandir al ancho real de la
            # silueta en esa Y, pero limitado al rango hombro-hombro con margen
            cx = (x_sh_min + x_sh_max) // 2
            half_w = max(
                (x_sh_max - x_sh_min) // 2,        # al menos ancho de hombros
                (x2_raw - x1_raw) // 2 - 10        # o silueta real menos brazos
            )
            x1 = max(x1_raw, cx - half_w)
            x2 = min(x2_raw, cx + half_w)

        elif zone_name == "cadera":
            x_hi_izq = _lm_x(landmarks, 23, W, True)
            x_hi_der = _lm_x(landmarks, 24, W, True)
            cx   = (x_hi_izq + x_hi_der) // 2
            # Ancho: usar el contorno real de la silueta centrado en cx
            half_w = (x2_raw - x1_raw) // 2
            x1 = max(0, cx - half_w)
            x2 = min(W, cx + half_w)

        else:
            x1, x2 = x1_raw, x2_raw

        # Garantizar mínimo de 30px
        if x2 - x1 < 30:
            mid = (x1_raw + x2_raw) // 2
            x1, x2 = max(0, mid - 40), min(W, mid + 40)

        return {
            "y_center":  y_c,
            "rows":      _band(y_c, H),
            "width_pct": 100.0,
            "x1": x1, "x2": x2,
        }

    zones = {
        "cuello":    zone_from_mask(y_cuello,  "cuello"),
        "pecho":     zone_from_mask(y_pecho,   "pecho"),
        "cintura":   zone_from_mask(y_cintura, "cintura"),
        "cadera":    zone_from_mask(y_cadera,  "cadera"),
        "brazo_izq": _arm_zone(11, 13),
        "brazo_der": _arm_zone(12, 14),
    }

    # ── Muslo y rodilla, cada pierna por separado ─────────────────────────
    # Y desde landmarks (cadera 23/24, rodilla 25/26); X = tramo continuo de
    # silueta de ESA pierna, separado de la otra por la línea media entre
    # ambas caderas.
    x_h_izq = _lm_x(landmarks, 23, W, p); x_h_der = _lm_x(landmarks, 24, W, p)
    x_split = (x_h_izq + x_h_der) // 2
    leg_half = max(6, int(round(18 * W / 640.0)))

    def _leg_zone(hip_idx, knee_idx, kind, is_izq):
        yh = _lm_y(landmarks, hip_idx, H, p)
        yk = _lm_y(landmarks, knee_idx, H, p)
        if yk <= yh + 10:                       # rodilla mal detectada
            yk = min(H - 1, yh + int(0.25 * H))
        if kind == "muslo":
            y_c = yh + int(0.35 * (yk - yh))
            x_c = _lm_x(landmarks, hip_idx, W, p)
        else:
            y_c = yk
            x_c = _lm_x(landmarks, knee_idx, W, p)
        # lado de la imagen que le toca a esta pierna
        izq_en_izquierda = x_h_izq < x_h_der
        left_side = (is_izq == izq_en_izquierda)
        x_lo, x_hi = (0, x_split) if left_side else (x_split, W)
        run = _run_bounds(mask, y_c, x_c, x_lo, x_hi)
        if run is None or run[1] - run[0] < 6:
            run = (max(0, x_c - leg_half), min(W, x_c + leg_half))
        return {
            "y_center":  y_c,
            "rows":      _band(y_c, H),
            "width_pct": 100.0,
            "x1": run[0], "x2": run[1],
        }

    zones["muslo_izq"]   = _leg_zone(23, 25, "muslo",   True)
    zones["muslo_der"]   = _leg_zone(24, 26, "muslo",   False)
    zones["rodilla_izq"] = _leg_zone(23, 25, "rodilla", True)
    zones["rodilla_der"] = _leg_zone(24, 26, "rodilla", False)

    return zones


def _auto_zones_lateral(
    landmarks: list, H: int, W: int,
    y_pecho_ref: int, y_cintura_ref: int,
    mask: np.ndarray = None,
    side: str = "izq",
) -> dict:
    """
    Calcula zonas automáticas para vistas laterales.

    Zonas calculadas:
        prof_pecho   → misma Y que el pecho frontal
        prof_cintura → misma Y que la cintura frontal
        muslo        → 30% entre cadera (23/24) y rodilla (25/26)
        rodilla      → landmarks 25/26

    Args:
        y_pecho_ref:   Y del pecho detectado en la vista frontal
        y_cintura_ref: Y de la cintura detectada en la vista frontal

    Returns:
        dict {zona: {"y_center", "rows", "width_pct"}}
    """
    p = True  # apply_parallax
    y_hip   = (_lm_y(landmarks, 23, H, p) + _lm_y(landmarks, 24, H, p)) // 2
    y_knee  = (_lm_y(landmarks, 25, H, p) + _lm_y(landmarks, 26, H, p)) // 2

    # Calcular y_bottom real de la silueta para usar como referencia
    # cuando los landmarks de rodilla están fuera o mal detectados
    if mask is not None:
        rows_with_body = np.where(mask.sum(axis=1) > 10)[0]
        y_bottom = int(rows_with_body.max()) if len(rows_with_body) > 0 else H
    else:
        y_bottom = H

    # Si y_knee está muy cerca de y_bottom o más allá, usar y_bottom como referencia
    # Esto ocurre cuando la rodilla no es visible en la vista lateral
    if y_knee > y_bottom - 20 or y_knee <= y_hip:
        # Estimar rodilla como 60% entre cadera y fondo de silueta
        y_knee = y_hip + int(0.60 * (y_bottom - y_hip))

    leg_h = max(y_knee - y_hip, 50)  # mínimo 50px

    def zone(y_c, width_pct=100.0, band_x=20):
        # Usar _silhouette_x_bounds para obtener el ancho real de la silueta
        x1, x2 = 0, W
        if mask is not None:
            x1, x2 = _silhouette_x_bounds(mask, y_c, band=band_x)
        # Asegurar mínimo de 40px
        if x2 - x1 < 40:
            mid = (x1 + x2) // 2
            x1, x2 = max(0, mid-50), min(W, mid+50)
        return {
            "y_center":  y_c,
            "rows":      _band(y_c, H),
            "width_pct": width_pct,
            "x1": x1,
            "x2": x2,
        }

    # Estrategia: usar y_bottom como referencia principal.
    # Si la silueta está cortada (y_bottom muy cercano a y_knee),
    # extrapolar la pierna usando leg_h como estimación de la mitad.
    rango_total = y_bottom - y_hip

    # Silueta completa: y_bottom >> y_knee (piernas visibles)
    # Silueta cortada:  y_bottom ~= y_knee (solo hasta la rodilla)
    silueta_completa = (y_bottom - y_knee) > (leg_h * 0.5)

    # Rodilla = landmark 25/26 + 70px hacia abajo
    # Muslo = punto medio entre cadera y rodilla + 100px hacia abajo
    y_muslo   = ((y_hip + y_knee) // 2) + 100
    y_rodilla = y_knee + 70

    # Para muslo y rodilla, centrar en el landmark X además de usar silueta
    x_knee = (_lm_x(landmarks, 25, W, True) + _lm_x(landmarks, 26, W, True)) // 2
    x_hip  = (_lm_x(landmarks, 23, W, True) + _lm_x(landmarks, 24, W, True)) // 2

    def zone_knee(y_c):
        """Rodilla: cuadrante pequeño centrado en landmark 25/26."""
        half = 35
        return {
            "y_center":  y_c,
            "rows":      _band(y_c, H),
            "width_pct": 100.0,
            "x1": max(0, x_knee - half),
            "x2": min(W, x_knee + half),
        }

    def zone_thigh(y_c):
        """Muslo: ancho desde silueta pero centrado en landmark."""
        x1, x2 = 0, W
        if mask is not None:
            x1, x2 = _silhouette_x_bounds(mask, y_c, band=15)
        return {
            "y_center":  y_c,
            "rows":      _band(y_c, H),
            "width_pct": 100.0,
            "x1": x1, "x2": x2,
        }

    zones = {
        "prof_pecho":   zone(y_pecho_ref,   band_x=25),
        "prof_cintura": zone(y_cintura_ref, band_x=25),
        f"muslo_{side}":   zone_thigh(y_muslo),
        f"rodilla_{side}": zone_knee(y_rodilla),
    }

    # Brazo del lado que da a la cámara (grosor del brazo en la vista lateral)
    sh_i, el_i = (11, 13) if side == "izq" else (12, 14)
    y_sh_l = _lm_y(landmarks, sh_i, H, True)
    y_el_l = _lm_y(landmarks, el_i, H, True)
    x_sh_l = _lm_x(landmarks, sh_i, W, True)
    x_el_l = _lm_x(landmarks, el_i, W, True)
    y_br   = (y_sh_l + y_el_l) // 2
    x_br   = (x_sh_l + x_el_l) // 2
    half_b = max(6, int(round(22 * W / 1280.0)))
    zones[f"brazo_{side}"] = {
        "y_center":  y_br,
        "rows":      _band(y_br, H),
        "width_pct": 100.0,
        "x1": max(0, x_br - half_b), "x2": min(W, x_br + half_b),
        "orient": "v",
    }

    return zones


# ── Funciones de proceso ───────────────────────────────────────────────────────

def process_single_view(
    rgb: np.ndarray,
    depth: np.ndarray,
    view_name: str,
) -> dict:
    """Procesa una vista individual con el pipeline original."""
    cfg = VIEW_CONFIG[view_name]

    depth_clean = preprocess_depth(depth)
    seg = segment_body(
        rgb, depth_clean,
        d_min_mm=cfg["d_min"], d_max_mm=cfg["d_max"],
        x_min=cfg["x_min"],    x_max=cfg["x_max"],
        y_min=cfg["y_min"],    y_max=cfg["y_max"],
    )
    pcd = reconstruct_pointcloud(seg["depth_body"], seg["rgb_body"])

    mask = seg["mask"]
    rows_with_body = np.where(mask.sum(axis=1) > 10)[0]
    y_top    = int(rows_with_body.min()) if len(rows_with_body) > 0 else cfg["y_min"]
    y_bottom = int(rows_with_body.max()) if len(rows_with_body) > 0 else cfg["y_max"]

    return {
        "name":       view_name,
        "seg":        seg,
        "pcd":        pcd,
        "mask":       mask,
        "depth_body": seg["depth_body"],
        "rgb_body":   seg["rgb_body"],
        "rgb":        rgb,
        "y_top":      y_top,
        "y_bottom":   y_bottom,
        "height_px":  y_bottom - y_top,
    }


def extract_zone_at_norm(
    mask, depth_body, y_top, y_bottom, y_norm, zone_name, band=5
):
    """Mide ancho y profundidad en una posición normalizada de la silueta."""
    y_world = int(y_top + y_norm * (y_bottom - y_top))
    y_start = max(0, y_world - band)
    y_end   = min(mask.shape[0] - 1, y_world + band)
    return measure_row_range(
        mask, depth_body, y_start, y_end,
        width_percentile=ZONE_WIDTH_PCT.get(_base_zone(zone_name), 100.0),
    )


def measure_zone_rect(mask, depth_body, x1, y1, x2, y2):
    """Mide ancho y profundidad dentro de un rectángulo."""
    widths_px = []; depths_ref = []; deltas = []
    for y in range(y1, min(y2 + 1, mask.shape[0])):
        row_mask  = mask[y, x1:x2]
        row_depth = depth_body[y, x1:x2]
        active    = row_mask == 255
        if active.sum() < 4:
            continue
        d_valid = row_depth[active]
        d_valid = d_valid[d_valid > 0]
        if len(d_valid) < 3:
            continue
        widths_px.append(int(active.sum()))
        depths_ref.append(float(np.percentile(d_valid, 50)))
        deltas.append(float(np.percentile(d_valid, 90) -
                            np.percentile(d_valid, 10)))
    if not widths_px:
        return None
    width_px  = float(np.mean(widths_px))
    depth_ref = float(np.mean(depths_ref))
    delta_mm  = float(np.mean(deltas))
    return {"width_px": width_px,
            "width_mm": pixels_to_mm_width(width_px, depth_ref),
            "depth_mm": depth_ref, "delta_mm": delta_mm}


def measure_arm_rect(mask, depth_body, x1, y1, x2, y2, orient=None):
    """
    Mide el ANCHO de un brazo dentro de un rectángulo.

    · orient "v" (brazo colgando): se cuentan píxeles activos por FILA.
    · orient "h" (brazo extendido, p.ej. pose en T): el grosor del brazo
      es vertical, así que se cuentan píxeles activos por COLUMNA.
    Si orient es None se deduce de la forma del rectángulo (más ancho que
    alto → brazo horizontal).
    """
    if orient is None:
        orient = "h" if (x2 - x1) > 1.3 * (y2 - y1) else "v"
    if orient == "h":
        # Transponer: las columnas pasan a ser filas
        return measure_zone_rect(mask.T, depth_body.T, y1, x1, y2, x2)
    return measure_zone_rect(mask, depth_body, x1, y1, x2, y2)


def measure_symmetric_zone(mask, depth_body, x1, y1, x2, y2,
                            img_center_x=320):
    """Mide una zona y su simétrica."""
    d1     = measure_zone_rect(mask, depth_body, x1, y1, x2, y2)
    zone_w = x2 - x1
    mir_cx = 2 * img_center_x - (x1 + x2) // 2
    mx1 = max(0, mir_cx - zone_w // 2)
    mx2 = min(mask.shape[1] - 1, mir_cx + zone_w // 2)
    d2  = measure_zone_rect(mask, depth_body, mx1, y1, mx2, y2)
    if d1 and d2:
        return {k: (d1[k] + d2[k]) / 2 for k in d1}
    return d1 or d2


def _ellipse_cm(width_mm, depth_mm):
    return compute_circumference(width_mm, depth_mm) / 10.0


# ── Función principal ──────────────────────────────────────────────────────────

def combine_measurements(
    frontal_data:     dict,
    posterior_data:   dict,
    lateral_izq_data: dict,
    lateral_der_data: dict,
    rgb_image:        Optional[np.ndarray] = None,
    rgb_posterior:    Optional[np.ndarray] = None,
    rgb_lateral_izq:  Optional[np.ndarray] = None,
    rgb_lateral_der:  Optional[np.ndarray] = None,
) -> dict:
    """
    Combina medidas de las 4 vistas.

    Prioridad por zona:
        1. Zona marcada manualmente en VIEW_ZONES (override del usuario)
        2. Detección automática con MediaPipe
        3. Fallback: posición normalizada anatómica

    Args:
        frontal_data, posterior_data, lateral_*_data:
            dict con mask, depth_body, y_top, y_bottom
        rgb_image:       RGB frontal (BGR) — para MediaPipe frontal/posterior
        rgb_posterior:   RGB posterior (BGR)
        rgb_lateral_izq: RGB lateral izquierdo (BGR)
        rgb_lateral_der: RGB lateral derecho (BGR)

    Returns:
        dict por zona con: width_mm, depth_mm, circumference_mm,
                           circumference_cm, y_px
    """
    results = {}
    print("\n  Medidas antropométricas:")
    print("  " + "-" * 58)

    views = {
        "frontal":     frontal_data,
        "posterior":   posterior_data,
        "lateral_izq": lateral_izq_data,
        "lateral_der": lateral_der_data,
    }
    rgbs = {
        "frontal":     rgb_image,
        "posterior":   rgb_posterior,
        "lateral_izq": rgb_lateral_izq,
        "lateral_der": rgb_lateral_der,
    }

    # ── Detectar landmarks en todas las vistas disponibles ────────────────────
    print("\n  Detectando landmarks en todas las vistas...")
    landmarks_per_view = {}
    for vname, rgb in rgbs.items():
        if rgb is not None:
            lm = _detect_landmarks_for_view(rgb)
            if lm:
                H, W = rgb.shape[:2]
                landmarks_per_view[vname] = {"lm": lm, "H": H, "W": W}
                print(f"  ✓ {vname}: landmarks detectados")
            else:
                print(f"  ⚠ {vname}: no se detectaron landmarks")

    # ── Calcular zonas automáticas por vista ──────────────────────────────────
    auto_zones = {}  # {view: {zone: cfg}}

    # Frontal y posterior
    for vname in ("frontal", "posterior"):
        if vname in landmarks_per_view:
            d    = landmarks_per_view[vname]
            msk  = views[vname].get("mask")
            zns  = _auto_zones_frontal_posterior(
                d["lm"], d["H"], d["W"], mask=msk
            )
            # Posterior no necesita cuello — no es visible desde atrás
            if vname == "posterior":
                zns.pop("cuello", None)
            auto_zones[vname] = zns

    # Referencia Y pecho/cintura del frontal para las laterales
    y_pecho_ref   = None
    y_cintura_ref = None
    if "frontal" in auto_zones:
        y_pecho_ref   = auto_zones["frontal"].get("pecho",   {}).get("y_center")
        y_cintura_ref = auto_zones["frontal"].get("cintura", {}).get("y_center")

    # Laterales
    for vname in ("lateral_izq", "lateral_der"):
        if vname in landmarks_per_view and y_pecho_ref and y_cintura_ref:
            d   = landmarks_per_view[vname]
            msk = views[vname].get("mask")
            auto_zones[vname] = _auto_zones_lateral(
                d["lm"], d["H"], d["W"],
                y_pecho_ref, y_cintura_ref,
                mask=msk,
                side=("izq" if vname == "lateral_izq" else "der"),
            )

    # ── Helpers de medición ───────────────────────────────────────────────────

    def _measure_manual(view_name: str, zone: str) -> dict | None:
        """Mide usando el rectángulo marcado manualmente (prioridad alta)."""
        vd = views[view_name]
        if vd["mask"] is None:
            return None
        z = VIEW_ZONES.get(view_name, {}).get(zone)
        if not z:
            return None
        if _base_zone(zone) == "brazo":
            return measure_arm_rect(
                vd["mask"], vd["depth_body"],
                z["x1"], z["y1"], z["x2"], z["y2"]
            )
        return measure_zone_rect(
            vd["mask"], vd["depth_body"],
            z["x1"], z["y1"], z["x2"], z["y2"]
        )

    def _measure_auto(view_name: str, zone: str) -> dict | None:
        """Mide usando la zona detectada automáticamente por MediaPipe."""
        vd = views[view_name]
        if vd["mask"] is None:
            return None
        zone_cfg = auto_zones.get(view_name, {}).get(zone)
        if not zone_cfg:
            return None
        y_start, y_end = zone_cfg["rows"]
        width_pct      = zone_cfg.get("width_pct", 100.0)
        x1 = zone_cfg.get("x1", 0)
        x2 = zone_cfg.get("x2", vd["mask"].shape[1])
        # Si hay x1/x2 específicos (brazos), usar measure_zone_rect
        if x1 > 0 or x2 < vd["mask"].shape[1]:
            if _base_zone(zone) == "brazo":
                return measure_arm_rect(
                    vd["mask"], vd["depth_body"],
                    x1, y_start, x2, y_end,
                    orient=zone_cfg.get("orient"),
                )
            return measure_zone_rect(
                vd["mask"], vd["depth_body"],
                x1, y_start, x2, y_end
            )
        return measure_row_range(
            vd["mask"], vd["depth_body"],
            y_start, y_end,
            width_percentile=width_pct,
        )

    def _measure_fallback(view_name: str, zone: str) -> dict | None:
        """Mide usando posición normalizada anatómica (último recurso)."""
        vd = views[view_name]
        if vd["mask"] is None or rgbs.get(view_name) is None:
            return None     # vista no cargada (máscara de relleno)
        base = _base_zone(zone)
        # Sin landmarks no se puede aislar un solo brazo/pierna en vistas
        # frontal/posterior (la fila abarcaría ambos lados) ni el brazo en
        # lateral (se funde con el torso): sin dato.
        if base in ("brazo", "muslo", "rodilla"):
            if view_name in ("frontal", "posterior") or base == "brazo":
                return None
        y_norm = ANATOMICAL_NORM.get(base)
        if y_norm is None:
            return None
        return extract_zone_at_norm(
            vd["mask"], vd["depth_body"],
            vd["y_top"], vd["y_bottom"],
            y_norm, zone
        )

    def _measure(view_name: str, zone: str) -> dict | None:
        """Mide con prioridad: manual → auto → fallback."""
        return (_measure_manual(view_name, zone) or
                _measure_auto(view_name, zone) or
                _measure_fallback(view_name, zone))

    def _measure_sym(view_name: str, zone: str) -> dict | None:
        """Mide zona simétrica (muslo/rodilla)."""
        vd = views[view_name]
        if vd["mask"] is None:
            return None
        # Manual primero
        z = VIEW_ZONES.get(view_name, {}).get(zone)
        if z:
            return measure_symmetric_zone(
                vd["mask"], vd["depth_body"],
                z["x1"], z["y1"], z["x2"], z["y2"],
                img_center_x=IMG_CENTER_X,
            )
        # Auto
        zone_cfg = auto_zones.get(view_name, {}).get(zone)
        if zone_cfg:
            y_start, y_end = zone_cfg["rows"]
            return measure_row_range(
                vd["mask"], vd["depth_body"],
                y_start, y_end,
                width_percentile=zone_cfg.get("width_pct", 100.0),
            )
        return None

    def _avg(*dicts) -> dict | None:
        valid = [d for d in dicts if d is not None]
        if not valid:
            return None
        keys = valid[0].keys()
        return {k: float(np.mean([d[k] for d in valid])) for k in keys}

    def _get_y_px(zone: str, view: str = "frontal") -> Optional[int]:
        """Obtiene y_px de MediaPipe para el overlay del GUI."""
        return auto_zones.get(view, {}).get(zone, {}).get("y_center")

    def _store(zone: str, width_mm, depth_mm, delta_mm=None,
               y_px=None, view_name: str = "frontal",
               zone_override: str = None):
        # zone_override permite usar una zona diferente para obtener el rect
        # (ej: "brazo" usa "brazo_izq" como referencia de posición)
        _zone_for_rect = zone_override or zone
        if depth_mm is None and delta_mm is not None:
            depth_mm = delta_mm
        if width_mm and depth_mm and width_mm > 10 and depth_mm > 10:
            circ_mm = compute_circumference(width_mm, depth_mm)
            circ_cm = round(circ_mm / 10.0, 1)

            # Rectángulos para visualización: se buscan en TODAS las vistas
            # donde exista esta zona (marcada a mano o automática).
            def _rect_for(vn):
                mz = VIEW_ZONES.get(vn, {}).get(_zone_for_rect)
                if mz:
                    return {"x1": mz["x1"], "y1": mz["y1"],
                            "x2": mz["x2"], "y2": mz["y2"], "view": vn}
                cfg = auto_zones.get(vn, {}).get(_zone_for_rect)
                if cfg:
                    ys, ye = cfg["rows"]
                    return {"x1": cfg.get("x1", 0), "y1": ys,
                            "x2": cfg.get("x2", views[vn]["mask"].shape[1]
                                          if views[vn].get("mask") is not None
                                          else 640),
                            "y2": ye, "view": vn}
                return None

            all_rects = {}
            for vn in ("frontal", "posterior", "lateral_izq", "lateral_der"):
                r_ = _rect_for(vn)
                if r_:
                    all_rects[vn] = r_
            rect = all_rects.get(view_name) or (
                next(iter(all_rects.values())) if all_rects else None)

            results[zone] = {
                "width_mm":         round(width_mm, 1),
                "depth_mm":         round(depth_mm, 1),
                "delta_mm":         round(depth_mm, 1),
                "circumference_mm": round(circ_mm,  1),
                "circumference_cm": circ_cm,
                "y_px":             y_px,
                "rect":             rect,        # vista principal
                "all_rects":        all_rects,   # todas las vistas
            }
            print(f"  {zone:12s}: ancho={width_mm:.0f}mm  "
                  f"prof={depth_mm:.0f}mm  → {circ_cm:.1f} cm"
                  + (f"  (y={y_px}px)" if y_px else ""))
        else:
            results[zone] = None
            print(f"  {zone:12s}: -- (sin datos)")

    # ── Grosor (profundidad) desde las vistas laterales ──────────────────────
    def _lat_from_norm(lat_view: str, y_ref, base: str):
        """
        Si en la lateral no hay zona (landmarks que fallan en perfil, o no se
        marcó prof_pecho/prof_cintura), se mide el grosor de la silueta
        lateral a la MISMA altura relativa del cuerpo que en la frontal.
        """
        # Vista no cargada: la app pasa la máscara frontal como "relleno";
        # medirla daría el ANCHO frontal como si fuera grosor.
        if rgbs.get(lat_view) is None:
            return None
        lv = views.get(lat_view)
        fv = views.get("frontal")
        if lv is None or lv.get("mask") is None:
            return None
        y_norm = ANATOMICAL_NORM.get(base)
        try:
            if y_ref is not None and fv is not None and fv.get("mask") is not None:
                span = fv["y_bottom"] - fv["y_top"]
                if span > 50:
                    y_norm = (y_ref - fv["y_top"]) / span
        except Exception:
            pass
        if y_norm is None:
            return None
        return extract_zone_at_norm(
            lv["mask"], lv["depth_body"], lv["y_top"], lv["y_bottom"],
            float(np.clip(y_norm, 0.05, 0.95)), "prof_" + base, band=4)

    def _lat_thickness(prof_key: str, base: str, y_ref):
        vals = []
        for lat_view in ("lateral_izq", "lateral_der"):
            m = _measure(lat_view, prof_key)
            if m is None:
                m = _lat_from_norm(lat_view, y_ref, base)
                if m is not None:
                    print(f"    · {lat_view}: {prof_key} sin zona → grosor "
                          f"medido a altura relativa del frontal")
            if m is not None:
                vals.append(m)
        return _avg(*vals) if vals else None

    # ── Cuello ────────────────────────────────────────────────────────────────
    f_cuello = _measure("frontal", "cuello")
    lat_pecho = _lat_thickness("prof_pecho", "pecho", y_pecho_ref)
    # El grosor del CUELLO no sale de la lateral a la altura del pecho
    # (daría ~23 cm de profundidad para un cuello de ~11 cm de ancho):
    # se usa la relación anatómica.
    _cuello_depth = (f_cuello["width_mm"] * DEPTH_WIDTH_RATIO["cuello"]
                     if f_cuello else None)
    _store("cuello",
           f_cuello["width_mm"] if f_cuello else None,
           _cuello_depth,
           f_cuello["delta_mm"] if f_cuello else None,
           y_px=_get_y_px("cuello"), view_name="frontal")

    # ── Pecho ─────────────────────────────────────────────────────────────────
    f_pecho = _measure("frontal", "pecho")
    _store("pecho",
           f_pecho["width_mm"] if f_pecho else None,
           (lat_pecho["width_mm"] if lat_pecho else
            (f_pecho["width_mm"] * DEPTH_WIDTH_RATIO["pecho"]
             if f_pecho else None)),
           f_pecho["delta_mm"] if f_pecho else None,
           y_px=_get_y_px("pecho"), view_name="frontal")

    # ── Brazos, muslos y rodillas: izquierdo y derecho por separado ───────────
    # Ancho = promedio de frontal y posterior (ese lado).
    # Profundidad = grosor medido en la vista lateral de ese lado; si no hay
    # lateral se usa LIMB_DEPTH_RATIO × ancho.
    side_dep = {}
    for _base in ("brazo", "muslo", "rodilla"):
        for _side in ("izq", "der"):
            _key = f"{_base}_{_side}"
            _w = _avg(_measure("frontal", _key), _measure("posterior", _key))
            _lat = _measure(f"lateral_{_side}", _key)
            _wmm = _w["width_mm"] if _w else None
            _dmm = _lat["width_mm"] if _lat else (
                _wmm * LIMB_DEPTH_RATIO if _wmm else None)
            _store(_key, _wmm, _dmm,
                   _w["delta_mm"] if _w else None,
                   y_px=_get_y_px(_key), view_name="frontal")

    # ── Cintura ───────────────────────────────────────────────────────────────
    cin_f = _measure("frontal",   "cintura")
    cin_p = _measure("posterior", "cintura")
    cin   = _avg(cin_f, cin_p) or cin_f
    lat_cin = _lat_thickness("prof_cintura", "cintura", y_cintura_ref)
    _store("cintura",
           cin["width_mm"] if cin else None,
           (lat_cin["width_mm"] if lat_cin else
            (cin["width_mm"] * DEPTH_WIDTH_RATIO["cintura"]
             if cin else None)),
           cin["delta_mm"] if cin else None,
           y_px=_get_y_px("cintura"), view_name="frontal")

    # ── Cadera ────────────────────────────────────────────────────────────────
    cad_f = _measure("frontal",   "cadera")
    cad_p = _measure("posterior", "cadera")
    cad   = _avg(cad_f, cad_p) or cad_f
    _store("cadera",
           cad["width_mm"] if cad else None,
           (lat_cin["width_mm"] if lat_cin else
            (cad["width_mm"] * DEPTH_WIDTH_RATIO["cadera"]
             if cad else None)),
           cad["delta_mm"] if cad else None,
           y_px=_get_y_px("cadera"), view_name="frontal")

    # ── Valores globales (promedio de ambos lados) ────────────────────────────
    # Se conservan "brazo", "muslo" y "rodilla" como promedio izq/der porque
    # otras partes de la app (PDF, comparación) los usan; no llevan
    # rectángulos propios para no duplicar el dibujo en las imágenes.
    for _base in ("brazo", "muslo", "rodilla"):
        _l = results.get(f"{_base}_izq")
        _r = results.get(f"{_base}_der")
        _ok = [d for d in (_l, _r) if d]
        if not _ok:
            results[_base] = None
            continue
        _mean = lambda k: round(float(np.mean([d[k] for d in _ok])), 1)
        results[_base] = {
            "width_mm":         _mean("width_mm"),
            "depth_mm":         _mean("depth_mm"),
            "delta_mm":         _mean("delta_mm"),
            "circumference_mm": _mean("circumference_mm"),
            "circumference_cm": _mean("circumference_cm"),
            "y_px":             _ok[0].get("y_px"),
            "rect":             None,
            "all_rects":        {},
        }

    print("  " + "-" * 58)
    return results