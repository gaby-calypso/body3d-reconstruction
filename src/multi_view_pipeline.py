"""
multi_view_pipeline.py
-----------------------
Pipeline multi-vista usando el mismo código que funciona para vista única.
Aplica segment_body + reconstruct_pointcloud + measurements a cada vista
por separado, luego combina los resultados para el reporte.
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

# ── Zonas marcadas por vista desde la GUI ─────────────────────────────────────
# Estructura: {view_name: {zone_name: {x1, y1, x2, y2}}}
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

# ── Posiciones anatómicas normalizadas (0=arriba silueta, 1=abajo silueta) ───
ANATOMICAL_NORM = {
    "cuello":  0.10,
    "pecho":   0.28,
    "brazo":   0.32,
    "cintura": 0.48,
    "cadera":  0.58,
    "muslo":   0.72,
    "rodilla": 0.88,
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


def process_single_view(
    rgb: np.ndarray,
    depth: np.ndarray,
    view_name: str,
) -> dict:
    """
    Procesa una vista individual con el pipeline original.

    Returns dict con: seg, pcd, mask, depth_body, rgb_body,
                      y_top, y_bottom (límites de silueta)
    """
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
    mask: np.ndarray,
    depth_body: np.ndarray,
    y_top: int,
    y_bottom: int,
    y_norm: float,
    zone_name: str,
    band: int = 5,
) -> dict | None:
    """
    Mide ancho y profundidad en una posición normalizada de la silueta.

    Args:
        y_norm: posición normalizada [0=cabeza, 1=pies]
        band:   número de filas a promediar (±band)
    """
    y_world = int(y_top + y_norm * (y_bottom - y_top))
    y_start = max(0, y_world - band)
    y_end   = min(mask.shape[0] - 1, y_world + band)

    width_pct = ZONE_WIDTH_PCT.get(zone_name, 100.0)

    return measure_row_range(
        mask, depth_body,
        y_start, y_end,
        width_percentile=width_pct,
    )


def measure_zone_rect(
    mask: np.ndarray,
    depth_body: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> dict | None:
    """
    Mide ancho y profundidad dentro de un rectángulo marcado manualmente.
    """
    widths_px  = []
    depths_ref = []
    deltas     = []

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
    width_mm  = pixels_to_mm_width(width_px, depth_ref)
    return {"width_px": width_px, "width_mm": width_mm,
            "depth_mm": depth_ref, "delta_mm": delta_mm}


def measure_symmetric_zone(
    mask: np.ndarray,
    depth_body: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    img_center_x: int = 640,
) -> dict | None:
    """
    Mide una zona y su simétrica — para muslo/rodilla en vistas laterales.
    """
    d1 = measure_zone_rect(mask, depth_body, x1, y1, x2, y2)
    zone_w  = x2 - x1
    zone_cx = (x1 + x2) // 2
    mir_cx  = 2 * img_center_x - zone_cx
    mx1 = max(0, mir_cx - zone_w // 2)
    mx2 = min(mask.shape[1] - 1, mir_cx + zone_w // 2)
    d2 = measure_zone_rect(mask, depth_body, mx1, y1, mx2, y2)

    if d1 and d2:
        return {k: (d1[k] + d2[k]) / 2 for k in d1}
    return d1 or d2


def _ellipse_cm(width_mm: float, depth_mm: float) -> float:
    """Calcula perímetro de elipse en cm."""
    return compute_circumference(width_mm, depth_mm) / 10.0


def combine_measurements(
    frontal_data:     dict,
    posterior_data:   dict,
    lateral_izq_data: dict,
    lateral_der_data: dict,
    rgb_image:        Optional[np.ndarray] = None,   # ← nuevo parámetro
) -> dict:
    """
    Combina medidas de las 4 vistas usando zonas marcadas manualmente en la GUI.

    Si se provee rgb_image (imagen RGB frontal), se usa MediaPipe para detectar
    automáticamente las filas Y de cada zona. Si no, usa las zonas marcadas
    manualmente en VIEW_ZONES o el fallback calibrado de measurements.py.

    Estrategia:
      Frontal:   cuello, pecho, brazo_izq, brazo_der, cintura, cadera (ancho)
      Posterior: brazo_izq, brazo_der, cintura, cadera (ancho, promedia con frontal)
      Laterales: muslo, rodilla (ancho), prof_pecho, prof_cintura (profundidad)

    Args:
        frontal_data:     dict con mask, depth_body, y_top, y_bottom
        posterior_data:   dict con mask, depth_body, y_top, y_bottom
        lateral_izq_data: dict con mask, depth_body, y_top, y_bottom
        lateral_der_data: dict con mask, depth_body, y_top, y_bottom
        rgb_image:        imagen RGB frontal opcional (H, W, 3) — activa MediaPipe

    Returns:
        dict por zona con: width_mm, depth_mm, circumference_mm, circumference_cm, y_px
    """
    results = {}
    print("\n  Medidas antropométricas:")
    print("  " + "-" * 58)

    # ── Detectar zonas con MediaPipe (o fallback) ─────────────────────────────
    # build_zones() está en measurements.py:
    #   - Si rgb_image no es None → intenta MediaPipe → devuelve y_px por zona
    #   - Si MediaPipe falla o no está instalado → usa BODY_ZONES_FALLBACK
    # Resultado: {zona: {"y_center": int, "rows": (y_start, y_end), "width_pct": float}}
    print("\n  Detectando zonas de medición...")
    mediapipe_zones = build_zones(rgb_image)

    # ── Helpers internos ──────────────────────────────────────────────────────
    def _seg(view_name: str, zone: str) -> dict | None:
        """Mide una zona usando los rectángulos marcados manualmente en la GUI."""
        views = {
            "frontal":     frontal_data,
            "posterior":   posterior_data,
            "lateral_izq": lateral_izq_data,
            "lateral_der": lateral_der_data,
        }
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

    def _seg_auto(zone: str) -> dict | None:
        """
        Mide una zona frontal usando las filas detectadas por MediaPipe.

        Usa la banda (y_start, y_end) de mediapipe_zones para medir
        directamente en la máscara frontal, sin necesidad de rectángulo manual.
        """
        vd = frontal_data
        if vd["mask"] is None:
            return None
        zone_cfg = mediapipe_zones.get(zone)
        if not zone_cfg:
            return None
        y_start, y_end = zone_cfg["rows"]
        width_pct = zone_cfg.get("width_pct", 100.0)
        return measure_row_range(
            vd["mask"], vd["depth_body"],
            y_start, y_end,
            width_percentile=width_pct,
        )

    def _seg_sym(view_name: str, zone: str) -> dict | None:
        """Mide una zona simétrica (muslo/rodilla en laterales)."""
        views = {
            "frontal":     frontal_data,
            "posterior":   posterior_data,
            "lateral_izq": lateral_izq_data,
            "lateral_der": lateral_der_data,
        }
        vd = views[view_name]
        if vd["mask"] is None:
            return None
        z = VIEW_ZONES.get(view_name, {}).get(zone)
        if not z:
            return None
        return measure_symmetric_zone(
            vd["mask"], vd["depth_body"],
            z["x1"], z["y1"], z["x2"], z["y2"],
            img_center_x=IMG_CENTER_X,
        )

    def _avg(*dicts) -> dict | None:
        """Promedia varios resultados de medición."""
        valid = [d for d in dicts if d is not None]
        if not valid:
            return None
        keys = valid[0].keys()
        return {k: float(np.mean([d[k] for d in valid])) for k in keys}

    def _store(zone: str, width_mm: float | None,
               depth_mm: float | None, delta_mm: float | None = None):
        """
        Calcula perímetro, obtiene y_px de MediaPipe y guarda resultado.

        El y_px se incluye en el resultado para que el GUI pueda dibujar
        la línea de medición en la posición correcta de la imagen.
        """
        if depth_mm is None and delta_mm is not None:
            depth_mm = delta_mm

        # Obtener y_px de MediaPipe para esta zona (para el overlay del GUI)
        y_px = None
        zone_cfg = mediapipe_zones.get(zone)
        if zone_cfg:
            y_px = zone_cfg.get("y_center")

        if width_mm and depth_mm and width_mm > 10 and depth_mm > 10:
            circ_mm = compute_circumference(width_mm, depth_mm)
            circ_cm = round(circ_mm / 10.0, 1)
            results[zone] = {
                "width_mm":         round(width_mm, 1),
                "depth_mm":         round(depth_mm, 1),
                "delta_mm":         round(depth_mm, 1),
                "circumference_mm": round(circ_mm,  1),
                "circumference_cm": circ_cm,
                "y_px":             y_px,            # ← para líneas en el GUI
            }
            print(f"  {zone:12s}: ancho={width_mm:.0f}mm  "
                  f"prof={depth_mm:.0f}mm  → {circ_cm:.1f} cm"
                  + (f"  (y={y_px}px)" if y_px else ""))
        else:
            results[zone] = None
            print(f"  {zone:12s}: -- (sin datos)")

    # ── Cuello ────────────────────────────────────────────────────────────────
    # Prioridad: zona manual GUI → MediaPipe automático → fallback
    f_cuello = _seg("frontal", "cuello") or _seg_auto("cuello")
    lat_pecho = _avg(
        _seg("lateral_izq", "prof_pecho"),
        _seg("lateral_der", "prof_pecho"),
    )
    _store("cuello",
           f_cuello["width_mm"] if f_cuello else None,
           lat_pecho["width_mm"] if lat_pecho else None,
           f_cuello["delta_mm"] if f_cuello else None)

    # ── Pecho ─────────────────────────────────────────────────────────────────
    f_pecho = _seg("frontal", "pecho") or _seg_auto("pecho")
    _store("pecho",
           f_pecho["width_mm"] if f_pecho else None,
           lat_pecho["width_mm"] if lat_pecho else None,
           f_pecho["delta_mm"] if f_pecho else None)

    # ── Brazos ────────────────────────────────────────────────────────────────
    f_bi = _seg("frontal",   "brazo_izq")
    f_bd = _seg("frontal",   "brazo_der")
    p_bi = _seg("posterior", "brazo_izq")
    p_bd = _seg("posterior", "brazo_der")
    brazo = _avg(f_bi, f_bd, p_bi, p_bd)
    _store("brazo",
           brazo["width_mm"] if brazo else None,
           None,
           brazo["delta_mm"] if brazo else None)

    # ── Cintura ───────────────────────────────────────────────────────────────
    f_cin = _seg("frontal",   "cintura") or _seg_auto("cintura")
    p_cin = _seg("posterior", "cintura")
    cin   = _avg(f_cin, p_cin) or f_cin
    lat_cin = _avg(
        _seg("lateral_izq", "prof_cintura"),
        _seg("lateral_der", "prof_cintura"),
    )
    _store("cintura",
           cin["width_mm"] if cin else None,
           lat_cin["width_mm"] if lat_cin else None,
           cin["delta_mm"] if cin else None)

    # ── Cadera ────────────────────────────────────────────────────────────────
    f_cad = _seg("frontal",   "cadera") or _seg_auto("cadera")
    p_cad = _seg("posterior", "cadera")
    cad   = _avg(f_cad, p_cad) or f_cad
    _store("cadera",
           cad["width_mm"] if cad else None,
           lat_cin["width_mm"] if lat_cin else None,
           cad["delta_mm"] if cad else None)

    # ── Muslo ─────────────────────────────────────────────────────────────────
    li_m = _seg_sym("lateral_izq", "muslo")
    ld_m = _seg_sym("lateral_der", "muslo")
    muslo = _avg(li_m, ld_m)
    _store("muslo",
           muslo["width_mm"] if muslo else None,
           None,
           muslo["delta_mm"] if muslo else None)

    # ── Rodilla ───────────────────────────────────────────────────────────────
    li_r = _seg_sym("lateral_izq", "rodilla")
    ld_r = _seg_sym("lateral_der", "rodilla")
    rod   = _avg(li_r, ld_r)
    _store("rodilla",
           rod["width_mm"] if rod else None,
           None,
           rod["delta_mm"] if rod else None)

    print("  " + "-" * 58)
    return results