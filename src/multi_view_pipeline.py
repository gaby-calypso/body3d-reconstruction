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
ZONES_PER_VIEW = {
    "frontal":     ["cuello", "pecho", "brazo_izq", "brazo_der", "cintura", "cadera"],
    "posterior":   ["brazo_izq", "brazo_der", "cintura", "cadera"],
    "lateral_izq": ["muslo", "rodilla", "prof_pecho", "prof_cintura"],
    "lateral_der": ["muslo", "rodilla", "prof_pecho", "prof_cintura"],
}

IMG_CENTER_X = 640
BAND         = 8   # ±px alrededor del landmark para promediar filas

# Offset de paralaje RGB → Depth (calibrado con calibrate_parallax.py)
# Los landmarks se detectan en RGB — hay que desplazarlos para que
# coincidan con las posiciones correctas en el depth
PARALLAX_X = 27    # píxeles horizontales (opuesto al valor de calibración)
PARALLAX_Y = -17   # píxeles verticales


# ── Configuración por vista ────────────────────────────────────────────────────
VIEW_CONFIG = {
    "frontal": {
        "d_min": 1410, "d_max": 1790,
        "x_min": 396,  "x_max": 755,
        "y_min": 0,    "y_max": 626,
    },
    "posterior": {
        "d_min": 1180, "d_max": 1720,
        "x_min": 409,  "x_max": 780,
        "y_min": 0,    "y_max": 626,
    },
    "lateral_izq": {
        "d_min": 1090, "d_max": 1490,
        "x_min": 448,  "x_max": 716,
        "y_min": 0,    "y_max": 669,
    },
    "lateral_der": {
        "d_min": 930,  "d_max": 1750,
        "x_min": 422,  "x_max": 729,
        "y_min": 0,    "y_max": 626,
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
    Detecta los 33 landmarks de MediaPipe en una imagen RGB.

    Args:
        rgb: imagen BGR (H, W, 3) — formato cv2.imread

    Returns:
        lista de 33 landmarks normalizados, o None si falla
    """
    try:
        from src.pose_overlay import _detect_landmarks
        import cv2 as _cv2
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
        y = max(0, min(H-1, y + PARALLAX_Y))
    return y


def _lm_x(landmarks: list, idx: int, W: int,
           apply_parallax: bool = False) -> int:
    """Convierte landmark normalizado a coordenada X en píxeles.
    Si apply_parallax=True aplica el offset RGB→Depth."""
    x = int(landmarks[idx].x * W)
    if apply_parallax:
        x = max(0, min(W-1, x + PARALLAX_X))
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
    brazo_half  = 22

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
        "brazo_izq": {
            "y_center":  y_brazo_izq,
            "rows":      _band(y_brazo_izq, H),
            "width_pct": 100.0,
            "x1": max(0, x_brazo_izq - brazo_half),
            "x2": min(W, x_brazo_izq + brazo_half),
        },
        "brazo_der": {
            "y_center":  y_brazo_der,
            "rows":      _band(y_brazo_der, H),
            "width_pct": 100.0,
            "x1": max(0, x_brazo_der - brazo_half),
            "x2": min(W, x_brazo_der + brazo_half),
        },
    }

    return zones


def _auto_zones_lateral(
    landmarks: list, H: int, W: int,
    y_pecho_ref: int, y_cintura_ref: int,
    mask: np.ndarray = None,
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
        "muslo":        zone_thigh(y_muslo),
        "rodilla":      zone_knee(y_rodilla),
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
        width_percentile=ZONE_WIDTH_PCT.get(zone_name, 100.0),
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


def measure_symmetric_zone(mask, depth_body, x1, y1, x2, y2,
                            img_center_x=640):
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
        if vd["mask"] is None:
            return None
        y_norm = ANATOMICAL_NORM.get(zone)
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

            # Obtener coordenadas del rectángulo para visualización
            # El cuadrante usa los límites reales de la silueta en esa banda Y
            rect = None
            manual_z = VIEW_ZONES.get(view_name, {}).get(_zone_for_rect)
            if manual_z:
                rect = {"x1": manual_z["x1"], "y1": manual_z["y1"],
                        "x2": manual_z["x2"], "y2": manual_z["y2"],
                        "view": view_name}
            else:
                zone_cfg = auto_zones.get(view_name, {}).get(_zone_for_rect)
                if zone_cfg:
                    y_start, y_end = zone_cfg["rows"]
                    # Usar x1/x2 calculados desde los landmarks en auto_zones
                    x1 = zone_cfg.get("x1", 0)
                    x2 = zone_cfg.get("x2", 1280)
                    rect = {"x1": x1, "y1": y_start,
                            "x2": x2, "y2": y_end,
                            "view": view_name}

            # Generar rects para TODAS las vistas donde existe la zona
            # (no solo la vista principal de medición)
            all_rects = {}
            if rect:
                all_rects[rect["view"]] = rect

            # Agregar rects de vistas adicionales para visualización
            extra_views = {
                "cuello":  ["posterior"],
                "pecho":   ["posterior"],
                "cintura": ["posterior"],
                "cadera":  ["posterior"],
                "brazo":   ["posterior"],
                "muslo":   ["lateral_der"],
                "rodilla": ["lateral_der"],
            }
            for extra_view in extra_views.get(zone, []):
                zone_key = _zone_for_rect
                # Para brazos buscar brazo_izq en la vista extra
                extra_cfg = auto_zones.get(extra_view, {}).get(zone_key)
                if extra_cfg:
                    ys, ye = extra_cfg["rows"]
                    ex1 = extra_cfg.get("x1", 0)
                    ex2 = extra_cfg.get("x2", 1280)
                    all_rects[extra_view] = {
                        "x1": ex1, "y1": ys,
                        "x2": ex2, "y2": ye,
                        "view": extra_view
                    }

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

    # ── Cuello ────────────────────────────────────────────────────────────────
    f_cuello = _measure("frontal", "cuello")
    lat_pecho = _avg(
        _measure("lateral_izq", "prof_pecho"),
        _measure("lateral_der", "prof_pecho"),
    )
    _store("cuello",
           f_cuello["width_mm"] if f_cuello else None,
           lat_pecho["width_mm"] if lat_pecho else None,
           f_cuello["delta_mm"] if f_cuello else None,
           y_px=_get_y_px("cuello"), view_name="frontal")

    # ── Pecho ─────────────────────────────────────────────────────────────────
    f_pecho = _measure("frontal", "pecho")
    _store("pecho",
           f_pecho["width_mm"] if f_pecho else None,
           lat_pecho["width_mm"] if lat_pecho else None,
           f_pecho["delta_mm"] if f_pecho else None,
           y_px=_get_y_px("pecho"), view_name="frontal")

    # ── Brazos ────────────────────────────────────────────────────────────────
    brazo = _avg(
        _measure("frontal",   "brazo_izq"),
        _measure("frontal",   "brazo_der"),
        _measure("posterior", "brazo_izq"),
        _measure("posterior", "brazo_der"),
    )
    # Para el brazo usar brazo_izq como zona de referencia para el rect
    _store("brazo",
           brazo["width_mm"] if brazo else None,
           None,
           brazo["delta_mm"] if brazo else None,
           y_px=_get_y_px("brazo_izq"), view_name="frontal",
           zone_override="brazo_izq")

    # ── Cintura ───────────────────────────────────────────────────────────────
    cin_f = _measure("frontal",   "cintura")
    cin_p = _measure("posterior", "cintura")
    cin   = _avg(cin_f, cin_p) or cin_f
    lat_cin = _avg(
        _measure("lateral_izq", "prof_cintura"),
        _measure("lateral_der", "prof_cintura"),
    )
    _store("cintura",
           cin["width_mm"] if cin else None,
           lat_cin["width_mm"] if lat_cin else None,
           cin["delta_mm"] if cin else None,
           y_px=_get_y_px("cintura"), view_name="frontal")

    # ── Cadera ────────────────────────────────────────────────────────────────
    cad_f = _measure("frontal",   "cadera")
    cad_p = _measure("posterior", "cadera")
    cad   = _avg(cad_f, cad_p) or cad_f
    _store("cadera",
           cad["width_mm"] if cad else None,
           lat_cin["width_mm"] if lat_cin else None,
           cad["delta_mm"] if cad else None,
           y_px=_get_y_px("cadera"), view_name="frontal")

    # ── Muslo ─────────────────────────────────────────────────────────────────
    muslo = _avg(
        _measure_sym("lateral_izq", "muslo"),
        _measure_sym("lateral_der", "muslo"),
    )
    _store("muslo",
           muslo["width_mm"] if muslo else None,
           None,
           muslo["delta_mm"] if muslo else None,
           y_px=_get_y_px("muslo", "lateral_izq"), view_name="lateral_izq")

    # ── Rodilla ───────────────────────────────────────────────────────────────
    rod = _avg(
        _measure_sym("lateral_izq", "rodilla"),
        _measure_sym("lateral_der", "rodilla"),
    )
    _store("rodilla",
           rod["width_mm"] if rod else None,
           None,
           rod["delta_mm"] if rod else None,
           y_px=_get_y_px("rodilla", "lateral_izq"), view_name="lateral_izq")

    print("  " + "-" * 58)
    return results