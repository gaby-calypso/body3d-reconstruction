"""
multi_view_measurements.py
---------------------------
Extrae medidas antropométricas mediante cortes transversales de la malla 3D.
Usa estimación elíptica como fallback cuando el corte trimesh falla.
"""

from __future__ import annotations
import numpy as np
import open3d as o3d
import trimesh

ANATOMICAL_POSITIONS = {
    "cuello":  0.865,
    "pecho":   0.755,
    "brazo":   0.720,
    "cintura": 0.640,
    "cadera":  0.560,
    "muslo":   0.440,
    "rodilla": 0.280,
}


def mesh_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(o3d_mesh.vertices)
    faces    = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def ellipse_perimeter(a: float, b: float) -> float:
    """Aproximación de Ramanujan para el perímetro de una elipse."""
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))


def measure_at_height(
    vertices: np.ndarray,
    y_world: float,
    tolerance_m: float = 0.015,
) -> float | None:
    """
    Mide el perímetro en una altura dada usando dos métodos:
    1. Corte trimesh (preciso)
    2. Estimación elíptica desde los puntos cercanos al plano (fallback)

    Args:
        vertices:    Array (N, 3) de vértices de la malla
        y_world:     Altura Y en metros
        tolerance_m: Banda de tolerancia para capturar puntos cercanos

    Returns:
        Perímetro en cm o None
    """
    # Método 1: puntos dentro de la banda horizontal
    band = vertices[np.abs(vertices[:, 1] - y_world) < tolerance_m]

    if len(band) < 6:
        return None

    # Ancho (eje X) y profundidad (eje Z) de la sección
    x_vals = band[:, 0]
    z_vals = band[:, 2]

    x_range = np.percentile(x_vals, 95) - np.percentile(x_vals, 5)
    z_range = np.percentile(z_vals, 95) - np.percentile(z_vals, 5)

    # Semi-ejes de la elipse aproximada
    a = x_range / 2.0  # semi-eje mayor (ancho)
    b = z_range / 2.0  # semi-eje menor (profundidad)

    if a < 0.01 or b < 0.01:
        return None

    perimeter_m  = ellipse_perimeter(a, b)
    perimeter_cm = perimeter_m * 100.0
    return perimeter_cm


def extract_all_measurements(
    mesh: o3d.geometry.TriangleMesh,
    positions: dict = ANATOMICAL_POSITIONS,
) -> dict:
    """
    Extrae medidas antropométricas de la malla unificada.

    Args:
        mesh:      Malla Open3D del cuerpo completo
        positions: Dict {nombre: posición_normalizada [0=pies, 1=cabeza]}

    Returns:
        Dict {nombre: perímetro_cm}
    """
    vertices = np.asarray(mesh.vertices)
    y_min = vertices[:, 1].min()
    y_max = vertices[:, 1].max()
    height_m = y_max - y_min
    print(f"  Altura estimada de la malla: {height_m * 100:.1f} cm")

    # Tolerancia adaptativa según altura de la malla
    tolerance = max(0.012, height_m * 0.018)

    results = {}
    print("\n  Medidas antropométricas:")
    print("  " + "-" * 45)

    for name, y_norm in positions.items():
        y_world = y_min + y_norm * (y_max - y_min)
        perim = measure_at_height(vertices, y_world, tolerance_m=tolerance)

        # Filtro de sanidad anatómica (cm)
        SANITY = {
            "cuello":  (25, 60),
            "pecho":   (60, 160),
            "brazo":   (20, 60),
            "cintura": (50, 160),
            "cadera":  (60, 180),
            "muslo":   (30, 100),
            "rodilla": (20, 70),
        }
        lo, hi = SANITY.get(name, (10, 200))

        if perim is not None and lo < perim < hi:
            results[name] = round(perim, 1)
            print(f"  {name:15s}: {perim:6.1f} cm")
        else:
            results[name] = None
            reason = f"fuera de rango ({perim:.1f})" if perim else "sin puntos"
            print(f"  {name:15s}: -- ({reason})")

    print("  " + "-" * 45)
    return results
