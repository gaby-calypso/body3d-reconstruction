"""
measurements.py
---------------
Extrae medidas antropométricas de la silueta de profundidad segmentada.

Estrategia:
    Para cada zona del cuerpo (cuello, pecho, cintura, cadera):
    1. Identificar la fila Y de la zona usando BODY_ZONES (calibradas manualmente)
    2. Medir el ancho W en mm usando proyección inversa pinhole
    3. Medir la profundidad D en mm usando percentiles robustos (p10-p90)
    4. Modelar la sección transversal como una elipse (a=W/2, b=D/2)
    5. Calcular el perímetro con la fórmula de Ramanujan

Nota:
    La visualización de landmarks MediaPipe sobre las imágenes se realiza
    en src/pose_overlay.py — este módulo solo se ocupa de las mediciones.

Limitación conocida:
    Con vista frontal única no tenemos la parte trasera del cuerpo.
    La elipse es una aproximación — la precisión depende de cuánto
    se parece la sección real del cuerpo a una elipse en esa zona.

Inputs:
    - mask:       np.ndarray (H, W) uint8 — silueta segmentada
    - depth_body: np.ndarray (H, W) float32 — depth solo de la persona en mm

Outputs:
    - dict con medidas en mm para cada zona del cuerpo
"""

import numpy as np
from typing import Optional


# ── Parámetros intrínsecos D455 ───────────────────────────────────────────────
FX = 674.42
FY = 649.46
CX = 640.00
CY = 360.00

# ── Zonas de medición calibradas manualmente ──────────────────────────────────
# Determinadas empíricamente del análisis de distribución de la silueta.
# rows:      (y_start, y_end) — banda de filas a promediar para robustez
# width_pct: fracción del ancho central a usar (excluye brazos pegados al cuerpo)
BODY_ZONES = {
    "cuello":  {"rows": (284, 292), "width_pct": 100.0},
    "pecho":   {"rows": (371, 381), "width_pct":  75.0},
    "cintura": {"rows": (444, 454), "width_pct": 100.0},
    "cadera":  {"rows": (483, 493), "width_pct":  80.0},
}


def build_zones(rgb_image: Optional[np.ndarray] = None) -> dict:
    """
    Devuelve el diccionario de zonas de medición en formato unificado.

    El parámetro rgb_image se acepta por compatibilidad con multi_view_pipeline
    pero no se usa aquí — los landmarks MediaPipe son responsabilidad
    exclusiva de src/pose_overlay.py.

    Args:
        rgb_image: ignorado (mantenido por compatibilidad de firma)

    Returns:
        dict {zona: {"y_center": int, "rows": (y_start, y_end), "width_pct": float}}
    """
    return {
        name: {
            "y_center":  (cfg["rows"][0] + cfg["rows"][1]) // 2,
            "rows":       cfg["rows"],
            "width_pct":  cfg["width_pct"],
        }
        for name, cfg in BODY_ZONES.items()
    }


# ── Proyección y geometría ────────────────────────────────────────────────────

def pixels_to_mm_width(width_px: int, depth_mm: float,
                        fx: float = FX) -> float:
    """
    Convierte un ancho en píxeles a milímetros reales.

    Usa la proyección pinhole inversa:
        width_mm = width_px * depth_mm / fx

    Args:
        width_px:  ancho medido en píxeles
        depth_mm:  profundidad de referencia en mm
        fx:        focal length horizontal en píxeles

    Returns:
        ancho en mm
    """
    return width_px * depth_mm / fx


def ramanujan_perimeter(a: float, b: float) -> float:
    """
    Calcula el perímetro de una elipse con la aproximación de Ramanujan.

    Error < 0.0004% para proporciones típicas del cuerpo humano.

    Fórmula:
        h = (a - b)² / (a + b)²
        C ≈ π(a + b)(1 + 3h / (10 + √(4 - 3h)))

    Args:
        a: semi-eje mayor en mm (mitad del ancho frontal)
        b: semi-eje menor en mm (mitad de la profundidad)

    Returns:
        perímetro aproximado en mm
    """
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return float(np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h))))


# ── Medición por fila ─────────────────────────────────────────────────────────

def measure_row_range(mask: np.ndarray,
                      depth_body: np.ndarray,
                      y_start: int,
                      y_end: int,
                      width_percentile: float = 100.0) -> Optional[dict]:
    """
    Mide el ancho y la profundidad promedio en un rango de filas.

    Promedia varias filas para reducir el efecto del ruido.
    Usa percentiles robustos para profundidad (p10-p90) y permite
    recortar el ancho lateralmente para excluir brazos (width_percentile).

    Args:
        mask:             máscara binaria (H, W) uint8
        depth_body:       depth segmentado (H, W) float32 en mm
        y_start:          fila inicial del rango
        y_end:            fila final del rango
        width_percentile: fracción del ancho central a usar (default 100%)

    Returns:
        dict con width_px, width_mm, depth_mm, delta_mm, best_y
        None si no hay píxeles válidos
    """
    widths_px  = []
    depths_ref = []
    deltas     = []
    best_y     = y_start
    best_count = 0

    for y in range(y_start, y_end + 1):
        row_mask  = mask[y, :] == 255
        row_depth = depth_body[y, row_mask]

        if row_mask.sum() < 5:
            continue

        cols = np.where(row_mask)[0]
        total_w = len(cols)

        if width_percentile < 100.0:
            margin = int(total_w * (1.0 - width_percentile / 100.0) / 2)
            margin = max(0, margin)
            cols_trimmed  = cols[margin: total_w - margin]
            depth_trimmed = row_depth[margin: total_w - margin]
        else:
            cols_trimmed  = cols
            depth_trimmed = row_depth

        if len(cols_trimmed) < 3:
            continue

        widths_px.append(len(cols_trimmed))

        d_near = float(np.percentile(depth_trimmed, 10))
        d_far  = float(np.percentile(depth_trimmed, 90))
        d_ref  = float(np.percentile(depth_trimmed, 50))

        depths_ref.append(d_ref)
        deltas.append(d_far - d_near)

        if total_w > best_count:
            best_count = total_w
            best_y = y

    if not widths_px:
        return None

    width_px  = float(np.mean(widths_px))
    depth_ref = float(np.mean(depths_ref))
    delta_mm  = float(np.mean(deltas))
    width_mm  = pixels_to_mm_width(width_px, depth_ref)

    return {
        "width_px":  width_px,
        "width_mm":  width_mm,
        "depth_mm":  depth_ref,
        "delta_mm":  delta_mm,
        "best_y":    best_y,
    }


def compute_circumference(width_mm: float, delta_mm: float) -> float:
    """
    Calcula la circunferencia modelando la sección como elipse.

    Args:
        width_mm:  ancho frontal en mm
        delta_mm:  profundidad estimada del cuerpo en mm

    Returns:
        circunferencia en mm
    """
    return ramanujan_perimeter(width_mm / 2.0, delta_mm / 2.0)


# ── Función principal ─────────────────────────────────────────────────────────

def extract_measurements(mask: np.ndarray,
                          depth_body: np.ndarray,
                          rgb_image: Optional[np.ndarray] = None,
                          zones: Optional[dict] = None) -> dict:
    """
    Función principal: extrae todas las medidas antropométricas.

    Usa las zonas de BODY_ZONES (calibradas manualmente).
    El parámetro rgb_image se acepta por compatibilidad pero no se usa.

    Args:
        mask:       máscara binaria (H, W) uint8
        depth_body: depth segmentado (H, W) float32 en mm
        rgb_image:  ignorado (compatibilidad de firma)
        zones:      override manual del dict de zonas (opcional)

    Returns:
        dict por zona con width_mm, delta_mm, circumference_mm, circumference_cm, y_px
    """
    if zones is None:
        zones = build_zones()

    results = {}

    for zone_name, config in zones.items():
        y_start, y_end = config["rows"]
        width_pct      = config.get("width_pct", 100.0)

        row_data = measure_row_range(
            mask, depth_body, y_start, y_end,
            width_percentile=width_pct
        )

        if row_data is None:
            print(f"  ⚠ {zone_name}: no hay datos en y={y_start}-{y_end}")
            continue

        circ_mm = compute_circumference(
            row_data["width_mm"], row_data["delta_mm"]
        )

        results[zone_name] = {
            "y_px":             config.get("y_center", row_data["best_y"]),
            "width_mm":         round(row_data["width_mm"], 1),
            "depth_mm":         round(row_data["depth_mm"], 1),
            "delta_mm":         round(row_data["delta_mm"], 1),
            "circumference_mm": round(circ_mm, 1),
            "circumference_cm": round(circ_mm / 10.0, 1),
        }

        print(f"  ✓ {zone_name:8s}: "
              f"ancho={row_data['width_mm']:.0f}mm  "
              f"grosor={row_data['delta_mm']:.0f}mm  "
              f"→ contorno={circ_mm/10:.1f}cm  "
              f"(y={config.get('y_center', row_data['best_y'])}px)")

    return results