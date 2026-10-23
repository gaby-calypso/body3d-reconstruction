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


try:
    from src.calibration import gain_for as _calib_gain
    from src.calibration import capture_reference as _capture_reference
except Exception:                      # ejecución fuera del paquete
    try:
        from calibration import gain_for as _calib_gain
        from calibration import capture_reference as _capture_reference
    except Exception:
        def _calib_gain(zone):
            return 1.0

        def _capture_reference(rgb):
            return None


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


# Tolerancia de profundidad para decidir si un píxel pertenece al cuerpo.
# Todo lo que esté a más de ±BODY_DEPTH_TOL_MM de la profundidad del eje del
# cuerpo (pared, mesa, objetos del fondo) se descarta aunque la máscara lo
# haya dejado pasar. Un torso tiene ~±150 mm de relieve; el fondo está a
# más de 500 mm.
BODY_DEPTH_TOL_MM = 220.0

# Semiancho del cuello como fracción del ancho entre hombros (landmarks).
# La silueta del cuello NO es medible con fiabilidad en una vista frontal:
# el cabello, la mandíbula y el trapecio ensanchan la silueta (hasta 1.5-2×
# el cuello real). El cuello real mide ≈ 0.31 × ancho de hombros, así que se
# acota la caja a esa proporción (centrada en el eje del cuerpo).
NECK_HALF_FRAC = 0.17

# Tolerancia de profundidad alrededor del eje de un brazo (un brazo es un
# cilindro de ~±45 mm; la "estela" de borde queda a 100-300 mm).
ARM_DEPTH_TOL_MM = 90.0

# Eje del brazo = hombro→muñeca (más estable que hombro→codo con brazos
# estirados, que es el protocolo de captura).
ARM_AXIS_FROM_WRIST = True

# Niveles del tronco como fracción de la distancia hombros→caderas.
PECHO_FRAC   = 0.35
CINTURA_FRAC = 0.55
CADERA_FRAC  = 0.15
TORSO_BAND   = 12     # ±filas que se promedian en pecho/cintura/cadera


def _body_run(mask, depth, y_center, x_center, half_max, band=BAND,
              max_gap=4):
    """
    Límites X [x1, x2) de la silueta del cuerpo en la fila y_center, aislando
    el tramo continuo que contiene al eje del cuerpo (x_center).

    A diferencia de min/max sobre la máscara (que se estira hasta brazos
    extendidos, fondo o ruido), aquí:
      1. se mira solo la ventana [x_center ± half_max] (límite anatómico
         derivado de los landmarks: ancho de hombros / caderas);
      2. si hay depth, se descartan los píxeles cuya profundidad se aleja
         más de BODY_DEPTH_TOL_MM de la del eje (pared/fondo);
      3. una columna cuenta como cuerpo si ≥ 50 % de las filas de la banda
         están ocupadas (robusto al ruido de borde);
      4. se rellenan huecos de ≤ max_gap px y se toma el tramo continuo que
         contiene al eje.
    Devuelve None si no hay silueta utilizable.
    """
    if mask is None:
        return None
    H, W = mask.shape[:2]
    y1 = max(0, int(y_center) - band)
    y2 = min(H - 1, int(y_center) + band)
    x_lo = max(0, int(x_center - half_max))
    x_hi = min(W, int(x_center + half_max))
    if x_hi - x_lo < 3:
        return None

    ok = mask[y1:y2 + 1, x_lo:x_hi] == 255
    if depth is not None:
        d = depth[y1:y2 + 1, x_lo:x_hi]
        c0 = max(0, int(x_center) - x_lo - max(3, int(half_max * 0.12)))
        c1 = min(x_hi - x_lo, int(x_center) - x_lo + max(3, int(half_max * 0.12)) + 1)
        core = d[:, c0:c1]
        core = core[(core > 0) & ok[:, c0:c1]]
        if core.size >= 5:
            ref = float(np.median(core))
            ok &= (d > 0) & (np.abs(d - ref) <= BODY_DEPTH_TOL_MM)

    occ = ok.mean(axis=0) >= 0.5
    if not occ.any():
        return None

    # cerrar huecos pequeños
    idx = np.where(occ)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            occ[a:b] = True
    idx = np.where(occ)[0]

    runs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            runs.append((start, a)); start = b
    runs.append((start, idx[-1]))

    xc = int(x_center) - x_lo
    def dist(r):
        return 0 if r[0] <= xc <= r[1] else min(abs(xc - r[0]), abs(xc - r[1]))
    best = min(runs, key=dist)
    if dist(best) > half_max * 0.35:          # el eje cae en el vacío
        return None
    return int(best[0] + x_lo), int(best[1] + x_lo + 1)


def _arm_thickness(mask, depth, p0, p1,
                   fracs=(0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75), reach=45):
    """
    Grosor del brazo (px) medido PERPENDICULAR a su eje hombro→codo, en
    varias posiciones del tramo, y profundidad mediana del corte.
    Es independiente de que el brazo esté horizontal, inclinado o colgando
    (contar píxeles por fila o columna sobreestima en brazos inclinados).
    Devuelve (grosor_px, profundidad_mm) o None.
    """
    if mask is None:
        return None
    H, W = mask.shape[:2]
    v = np.array([p1[0] - p0[0], p1[1] - p0[1]], float)
    L = float(np.hypot(*v))
    if L < 8:
        return None
    u = v / L
    n = np.array([-u[1], u[0]])
    ths, deps = [], []
    for f in fracs:
        c = np.array(p0, float) + f * v
        ts = np.arange(-reach, reach + 0.5, 0.5)
        xs = np.clip(np.round(c[0] + ts * n[0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(c[1] + ts * n[1]).astype(int), 0, H - 1)
        on = mask[ys, xs] == 255
        if depth is not None:
            # El borde de profundidad de un brazo delgado tiene una "estela"
            # de píxeles mezclados con el fondo (sombra del estéreo) que la
            # máscara por rango no descarta. Solo cuentan los píxeles cuya
            # profundidad es la del propio brazo (±ARM_DEPTH_TOL_MM).
            dd_ = depth[ys, xs]
            i_c = int(np.argmin(np.abs(ts)))
            dc_ = dd_[max(0, i_c - 6):i_c + 7]
            dc_ = dc_[dc_ > 0]
            if dc_.size >= 3:
                ref_ = float(np.median(dc_))
                on = on & (dd_ > 0) & (np.abs(dd_ - ref_) <= ARM_DEPTH_TOL_MM)
        i0 = int(np.argmin(np.abs(ts)))
        if not on[i0]:                         # el eje cae fuera: buscar el más cercano
            idx = np.where(on)[0]
            if idx.size == 0:
                continue
            i0 = int(idx[np.argmin(np.abs(idx - i0))])
        a = i0
        while a > 0 and on[a - 1]:
            a -= 1
        b = i0
        while b < len(on) - 1 and on[b + 1]:
            b += 1
        ths.append((b - a + 1) * 0.5)
        if depth is not None:
            dv = depth[ys[a:b + 1], xs[a:b + 1]]
            dv = dv[dv > 0]
            if dv.size:
                deps.append(float(np.median(dv)))
    if not ths:
        return None
    dep = float(np.median(deps)) if deps else None
    return float(np.median(ths)), dep


def _refine_body_mask(mask, depth, landmarks, H, W, tol_mm=BODY_DEPTH_TOL_MM):
    """
    Limpia la máscara con la profundidad del propio cuerpo.

    Si en Captura se eligió un rango de profundidad amplio, la máscara deja
    pasar la pared o el fondo detrás de la persona y las cajas (sobre todo en
    brazos, que no tienen torso al lado) se llenan de fondo. Aquí se toma la
    profundidad del tronco (mediana alrededor del centro hombros-caderas) y se
    descarta todo píxel a más de ±tol_mm de ella; luego se conserva el
    componente conexo que contiene al tronco.
    Devuelve (mask, depth) nuevos; si no se puede (sin depth / sin referencia),
    devuelve los originales.
    """
    try:
        import cv2 as _cv2
        if mask is None or depth is None:
            return mask, depth
        pts = [(_lm_x(landmarks, i, W, True), _lm_y(landmarks, i, H, True))
               for i in (11, 12, 23, 24)]
        cx = int(np.mean([q[0] for q in pts])); cy = int(np.mean([q[1] for q in pts]))
        r = max(6, int(0.04 * W))
        win = depth[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
        win = win[(win > 0)]
        if win.size < 20:
            return mask, depth
        ref = float(np.median(win))
        ok  = (mask == 255) & (depth > 0) & (np.abs(depth - ref) <= tol_mm)
        ok  = ok.astype(np.uint8) * 255
        n, lab = _cv2.connectedComponents(ok)
        if n <= 1:
            return mask, depth
        c = lab[min(H - 1, max(0, cy)), min(W - 1, max(0, cx))]
        if c == 0:
            sizes = [(lab == i).sum() for i in range(1, n)]
            c = 1 + int(np.argmax(sizes))
        new = ((lab == c).astype(np.uint8) * 255)
        # cierre leve para no perder píxeles sueltos de borde
        new = _cv2.morphologyEx(new, _cv2.MORPH_CLOSE,
                                _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3, 3)))
        new = _cv2.bitwise_and(new, mask)
        nd = np.where(new == 255, depth, 0).astype(depth.dtype)
        return new, nd
    except Exception as e:
        print(f"  ⚠ no se pudo refinar la máscara: {e}")
        return mask, depth


def _auto_zones_frontal_posterior(
    landmarks: list, H: int, W: int,
    mask: np.ndarray = None,
    depth: np.ndarray = None,
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
        mask:  máscara segmentada opcional — si se provee, los límites
               X vienen del contorno real en lugar de estimaciones
        depth: depth (mm) del cuerpo segmentado, opcional — permite
               descartar pared/fondo que la máscara haya dejado pasar

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

    y_pecho   = y_sh + int(PECHO_FRAC * torso)
    y_cintura = y_sh + int(CINTURA_FRAC * torso)
    y_cadera  = y_hi - int(CADERA_FRAC * torso)

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
        # El codo es el landmark más inestable del brazo (en 27/29 saltó casi
        # 40 px). Con brazos estirados el eje hombro→muñeca es mucho más
        # robusto: el "codo efectivo" es el punto medio hombro→muñeca.
        wr_idx = 15 if el_idx == 13 else 16
        xw_, yw_ = _lm_x(landmarks, wr_idx, W, p), _lm_y(landmarks, wr_idx, H, p)
        if ARM_AXIS_FROM_WRIST and np.hypot(xw_ - xs_, yw_ - ys_) > 40:
            xe_, ye_ = (xs_ + xw_) // 2, (ys_ + yw_) // 2
        xm, ym = int(xs_ + 0.55 * (xe_ - xs_)), int(ys_ + 0.55 * (ye_ - ys_))
        geom = {"arm_geom": ((xs_, ys_), (xe_, ye_))}
        if abs(xe_ - xs_) > 1.2 * abs(ye_ - ys_):          # extendido
            return {
                "y_center":  ym,
                "rows":      (max(0, ym - brazo_vhalf), min(H - 1, ym + brazo_vhalf)),
                "width_pct": 100.0,
                "x1": max(0, xm - brazo_half), "x2": min(W, xm + brazo_half),
                "orient": "h", **geom,
            }
        return {                                            # colgando
            "y_center":  ym,
            "rows":      _band(ym, H),
            "width_pct": 100.0,
            "x1": max(0, xm - brazo_half), "x2": min(W, xm + brazo_half),
            "orient": "v", **geom,
        }

    # Geometría de referencia (px) para acotar cada zona.
    sh_w   = max(20, x_sh_max - x_sh_min)               # ancho entre hombros
    x_h_i  = _lm_x(landmarks, 23, W, p)
    x_h_d  = _lm_x(landmarks, 24, W, p)
    x_sh_c = (x_sh_min + x_sh_max) / 2.0
    x_hi_c = (x_h_i + x_h_d) / 2.0

    # Semiancho máximo permitido por zona, en fracción del ancho de hombros.
    # Es solo una red de seguridad: el ancho real lo da la silueta.
    HALF_FRAC = {"cuello": NECK_HALF_FRAC, "pecho": 0.55, "cintura": 0.52, "cadera": 0.55}

    # ── Niveles anatómicos desde la SILUETA (no solo desde landmarks) ────────
    # Los landmarks de MediaPipe son centros articulares y "tiemblan" de una
    # captura a otra; la silueta da niveles mucho más estables:
    #   cuello  = fila más estrecha entre la cabeza y los hombros
    # (cintura y cadera se dejaron en niveles fijos respecto a hombros/caderas:
    #  buscar la fila más estrecha/ancha añadía ruido entre capturas.)
    def _axis_x(y):
        t = 0.0 if y_hi == y_sh else min(1.0, max(0.0, (y - y_sh) / float(y_hi - y_sh)))
        return int(round(x_sh_c + t * (x_hi_c - x_sh_c)))

    def _profile(y_a, y_b, frac):
        ys, ws = [], []
        for yy in range(max(0, y_a), min(H - 1, y_b) + 1):
            run = _body_run(mask, depth, yy, _axis_x(yy), frac * sh_w, band=1)
            if run is not None:
                ys.append(yy); ws.append(run[1] - run[0])
        if len(ys) < 5:
            return None, None
        ws = np.convolve(np.pad(np.array(ws, float), 3, mode="edge"),
                         np.ones(7) / 7.0, mode="valid")
        return np.array(ys), ws

    y_knee_m = (_lm_y(landmarks, 25, H, p) + _lm_y(landmarks, 26, H, p)) // 2
    if y_knee_m <= y_hi + 20:
        y_knee_m = min(H - 1, y_hi + int(0.30 * H))
    if mask is not None:
        ys_, ws_ = _profile(y_nariz + int(0.20 * (y_sh - y_nariz)),
                            y_sh - int(0.05 * (y_sh - y_nariz)), 0.62)
        if ys_ is not None:
            y_cuello = int(ys_[int(np.argmin(ws_))])
    ZONE_BAND = {"cuello": 3}

    def zone_from_mask(y_c, zone_name):
        """
        MediaPipe da la Y; la silueta (máscara + depth) da el X real.
        El eje del cuerpo en esa Y se interpola entre el centro de los
        hombros y el centro de las caderas, de modo que sigue la postura.
        """
        t   = 0.0 if y_hi == y_sh else (y_c - y_sh) / float(y_hi - y_sh)
        t   = min(1.0, max(0.0, t))
        x_c = int(round(x_sh_c + t * (x_hi_c - x_sh_c)))
        half_max = HALF_FRAC[zone_name] * sh_w

        run = _body_run(mask, depth, y_c, x_c, half_max)
        if run is None:                       # sin silueta fiable → ventana teórica
            run = (int(x_c - 0.8 * half_max), int(x_c + 0.8 * half_max))
        x1, x2 = max(0, run[0]), min(W, run[1])

        # Garantizar un mínimo de ancho (el cuello es legítimamente estrecho)
        if x2 - x1 < (12 if zone_name == "cuello" else 30):
            x1, x2 = max(0, x_c - 40), min(W, x_c + 40)

        return {
            "y_center":  y_c,
            "rows":      _band(y_c, H, ZONE_BAND.get(zone_name, TORSO_BAND)),
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
            # Justo por DEBAJO de la rodilla: a la altura del landmark las
            # piernas pegadas se separan y 2-3 px de error cambian el ancho
            # de ~38 a ~27 px (ruido de ±3 cm). Un poco más abajo el ancho
            # es estable.
            ya = _lm_y(landmarks, knee_idx + 2, H, p)
            if ya <= yk + 10:
                ya = yk + int(0.25 * H)
            y_c = min(H - 1, yk + int(0.12 * (ya - yk)))
            x_c = _lm_x(landmarks, knee_idx, W, p)
        # lado de la imagen que le toca a esta pierna
        izq_en_izquierda = x_h_izq < x_h_der
        left_side = (is_izq == izq_en_izquierda)
        x_lo, x_hi = (0, x_split) if left_side else (x_split, W)
        x_c = min(max(x_c, x_lo + 1), x_hi - 1)
        run = _body_run(mask, depth, y_c, x_c, 0.40 * sh_w)
        if run is not None:                    # no invadir la otra pierna
            run = (max(run[0], x_lo), min(run[1], x_hi))
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
    depth: np.ndarray = None,
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

    # Eje del cuerpo en la vista lateral y semiancho máximo (profundidad del
    # cuerpo ≈ 0.25·largo de torso; se deja margen).
    y_sh_lat = (_lm_y(landmarks, 11, H, True) + _lm_y(landmarks, 12, H, True)) // 2
    torso_lat = max(40, y_hip - y_sh_lat)
    x_axis = (_lm_x(landmarks, 23, W, True) + _lm_x(landmarks, 24, W, True)) // 2
    lat_half = 0.45 * torso_lat

    def _lat_bounds(y_c, band_x):
        run = _body_run(mask, depth, y_c, x_axis, lat_half, band=band_x)
        if run is None and mask is not None:
            run = _silhouette_x_bounds(mask, y_c, band=band_x)
        return run if run is not None else (0, W)

    def zone(y_c, width_pct=100.0, band_x=20):
        x1, x2 = _lat_bounds(y_c, band_x)
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
        x1, x2 = _lat_bounds(y_c, 15)
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
    # Compensación TEMPORAL por captura (ver src/calibration.py): si esta
    # captura tiene valores manuales conocidos, cada zona se fija a ellos.
    _cap_ref = _capture_reference(rgb_image)
    if _cap_ref:
        print("  ⚠ Captura con referencia manual: circunferencias "
              "compensadas (calibración preliminar, no es medición libre).")
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

    # ── Limpiar máscaras con la profundidad del propio cuerpo ─────────────────
    for vname, d in landmarks_per_view.items():
        vd = views.get(vname)
        if vd and vd.get("mask") is not None:
            m2, d2 = _refine_body_mask(vd["mask"], vd.get("depth_body"),
                                       d["lm"], d["H"], d["W"])
            views[vname] = {**vd, "mask": m2, "depth_body": d2}

    # ── Calcular zonas automáticas por vista ──────────────────────────────────
    auto_zones = {}  # {view: {zone: cfg}}

    # Frontal y posterior
    for vname in ("frontal", "posterior"):
        if vname in landmarks_per_view:
            d    = landmarks_per_view[vname]
            msk  = views[vname].get("mask")
            zns  = _auto_zones_frontal_posterior(
                d["lm"], d["H"], d["W"], mask=msk,
                depth=views[vname].get("depth_body"),
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
                depth=views[vname].get("depth_body"),
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
        if zone_cfg.get("arm_geom") and _base_zone(zone) == "brazo":
            _t = _arm_thickness(vd["mask"], vd.get("depth_body"),
                                *zone_cfg["arm_geom"])
            if _t and _t[1]:
                _wmm = pixels_to_mm_width(_t[0], _t[1])
                return {"width_px": _t[0], "width_mm": _wmm,
                        "depth_mm": _t[1], "delta_mm": _wmm * LIMB_DEPTH_RATIO}
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
            circ_raw_cm = round(circ_mm / 10.0, 1)
            # Compensación del sesgo sistemático de la detección automática
            # (src/calibration.py). No se aplica a zonas marcadas a mano.
            _manual = any(VIEW_ZONES.get(vn, {}).get(_zone_for_rect)
                          for vn in views)
            _gain = 1.0 if _manual else _calib_gain(zone)
            circ_mm = circ_mm * _gain
            circ_cm = round(circ_mm / 10.0, 1)
            circ_auto_cm = circ_cm
            _forced = bool(_cap_ref and not _manual and zone in _cap_ref)
            if _forced:
                circ_cm = float(_cap_ref[zone])
                circ_mm = circ_cm * 10.0

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
                "circumference_raw_cm": circ_raw_cm,   # sin compensación
                "calib_gain":       _gain,
                "circumference_auto_cm": circ_auto_cm,   # antes de compensar por captura
                "calibrado_con_referencia_manual": _forced,
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