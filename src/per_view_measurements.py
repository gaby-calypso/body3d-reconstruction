"""
per_view_measurements.py
-------------------------
Extrae medidas antropométricas usando las 4 mallas individuales.

Estrategia por vista:
  - Frontal/Posterior: ancho en X a cada altura Y → ancho del cuerpo
  - Lateral izq/der:   ancho en X a cada altura Y → profundidad del cuerpo
  - Perímetro estimado = elipse con semi-ejes (ancho_frontal/2, prof_lateral/2)
"""

from __future__ import annotations
import numpy as np
import open3d as o3d

ANATOMICAL_POSITIONS = {
    "cuello":  0.865,
    "pecho":   0.755,
    "brazo":   0.720,
    "cintura": 0.640,
    "cadera":  0.560,
    "muslo":   0.440,
    "rodilla": 0.280,
}

SANITY_CM = {
    "cuello":  (20,  70),
    "pecho":   (40, 180),
    "brazo":   (15,  80),
    "cintura": (40, 180),
    "cadera":  (40, 180),
    "muslo":   (20, 120),
    "rodilla": (15,  80),
}


def ellipse_perimeter(a: float, b: float) -> float:
    """Aproximación de Ramanujan."""
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-9)
    return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))


def get_width_at_height(
    vertices: np.ndarray,
    y_norm: float,
    y_min: float,
    y_max: float,
    tolerance: float = 0.025,
    axis: int = 0,
) -> float | None:
    """
    Mide el ancho de la nube de vértices a una altura normalizada.

    Args:
        vertices:  Array (N,3)
        y_norm:    Posición normalizada [0=abajo, 1=arriba]
        y_min/max: Rango Y de la malla
        tolerance: Banda en metros alrededor del plano de corte
        axis:      0=X (ancho), 2=Z (profundidad)

    Returns:
        Ancho en metros o None
    """
    y_world = y_min + y_norm * (y_max - y_min)
    band = vertices[np.abs(vertices[:, 1] - y_world) < tolerance]
    if len(band) < 8:
        return None
    vals = band[:, axis]
    # Usar percentil 20-80 para excluir artefactos de borde
    width = np.percentile(vals, 80) - np.percentile(vals, 20)
    return width if width > 0.01 else None


def extract_measurements_from_views(
    meshes: dict,
    positions: dict = ANATOMICAL_POSITIONS,
) -> dict:
    """
    Extrae medidas combinando las 4 vistas individuales.

    Args:
        meshes: Dict {view_name: o3d.geometry.TriangleMesh}
        positions: Posiciones anatómicas normalizadas

    Returns:
        Dict con medidas y datos de diagnóstico por zona
    """
    # Extraer vértices por vista
    verts = {name: np.asarray(m.vertices) for name, m in meshes.items()}

    # Rango Y de la vista frontal (referencia de altura)
    ref_verts = verts.get("frontal", list(verts.values())[0])
    y_min = ref_verts[:, 1].min()
    y_max = ref_verts[:, 1].max()
    height_m = y_max - y_min
    print(f"  Altura de referencia (frontal): {height_m*100:.1f} cm")

    results   = {}
    diag      = {}
    tolerance = max(0.018, height_m * 0.02)

    print("\n  Medidas antropométricas (elipse: ancho frontal × prof. lateral):")
    print("  " + "-" * 60)

    for name, y_norm in positions.items():
        # Ancho desde vista frontal (eje X)
        w_front = None
        for view in ["frontal", "posterior"]:
            if view in verts:
                w = get_width_at_height(verts[view], y_norm, y_min, y_max,
                                        tolerance=tolerance, axis=0)
                if w is not None:
                    w_front = w if w_front is None else (w_front + w) / 2

        # Profundidad desde vistas laterales (eje Z = profundidad del cuerpo)
        w_side = None
        for view in ["lateral_izq", "lateral_der"]:
            if view in verts:
                v = verts[view]
                y_min_v = v[:, 1].min()
                y_max_v = v[:, 1].max()
                # En vista lateral el cuerpo se ve de costado: Z es la profundidad
                w = get_width_at_height(v, y_norm, y_min_v, y_max_v,
                                        tolerance=tolerance, axis=2)
                if w is not None:
                    w_side = w if w_side is None else (w_side + w) / 2

        # Calcular perímetro
        if w_front is not None and w_side is not None:
            a = w_front / 2  # semi-eje ancho
            b = w_side  / 2  # semi-eje profundidad
            perim_cm = ellipse_perimeter(a, b) * 100
        elif w_front is not None:
            # Solo vista frontal: asumir sección circular
            a = b = w_front / 2
            perim_cm = ellipse_perimeter(a, b) * 100
        else:
            perim_cm = None

        lo, hi = SANITY_CM.get(name, (10, 200))
        if perim_cm is not None and lo < perim_cm < hi:
            results[name] = round(perim_cm, 1)
            wf_cm = f"{w_front*100:.1f}" if w_front else "--"
            ws_cm = f"{w_side*100:.1f}"  if w_side  else "--"
            print(f"  {name:12s}: {perim_cm:6.1f} cm  "
                  f"(ancho={wf_cm}cm  prof={ws_cm}cm)")
        else:
            results[name] = None
            reason = f"{perim_cm:.1f}cm fuera rango" if perim_cm else "sin datos"
            print(f"  {name:12s}: --  ({reason})")

        diag[name] = {
            "w_front_cm": round(w_front * 100, 1) if w_front else None,
            "w_side_cm":  round(w_side  * 100, 1) if w_side  else None,
            "perim_cm":   results[name],
        }

    print("  " + "-" * 60)
    return results, diag
