"""
reconstruction.py
-----------------
Convierte la silueta del mapa de profundidad en una nube de puntos 3D.

Principio matemático:
    Cada píxel (u, v) con profundidad Z se convierte a coordenadas 3D
    usando los parámetros intrínsecos de la cámara:

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = Z (profundidad directa)

    Esto es la proyección inversa del modelo de cámara pinhole.

Parámetros intrínsecos D455 (calculados del datasheet March 2022):
    Depth FOV HD 1280x720: H=87°, V=58°
    fx = 674.42 px, fy = 649.46 px
    cx = 640.00 px, cy = 360.00 px

Inputs:
    - depth_body: np.ndarray (H, W) float32 — depth solo de la persona en mm
    - rgb_body:   np.ndarray (H, W, 3) uint8 — RGB opcional para colorear

Outputs:
    - points:  np.ndarray (N, 3) float32 — nube de puntos en mm (X, Y, Z)
    - colors:  np.ndarray (N, 3) float32 — colores RGB normalizados 0-1
"""

import numpy as np
import open3d as o3d


# ── Parámetros intrínsecos D455 (datasheet March 2022, depth HD 1280x720) ────
FX = 674.42   # focal length horizontal en píxeles
FY = 649.46   # focal length vertical en píxeles
CX = 640.00   # centro óptico horizontal (cx) en píxeles
CY = 360.00   # centro óptico vertical (cy) en píxeles
DEPTH_SCALE = 1.0  # los valores ya están en mm, sin conversión necesaria


def depth_to_pointcloud(depth_body: np.ndarray,
                         rgb_body: np.ndarray,
                         fx: float = FX,
                         fy: float = FY,
                         cx: float = CX,
                         cy: float = CY) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte un mapa de profundidad en una nube de puntos 3D.

    Para cada píxel válido (depth > 0) aplica la proyección inversa
    del modelo pinhole para obtener sus coordenadas 3D reales.

    Args:
        depth_body: mapa de profundidad (H, W) float32 en mm
        rgb_body:   imagen RGB (H, W, 3) uint8
        fx, fy:     focal lengths en píxeles
        cx, cy:     centro óptico en píxeles

    Returns:
        points: np.ndarray (N, 3) float32 — coordenadas X, Y, Z en mm
        colors: np.ndarray (N, 3) float32 — colores RGB normalizados 0-1
    """
    H, W = depth_body.shape

    # Crear grilla de coordenadas de píxeles
    u_coords = np.arange(W, dtype=np.float32)  # columnas
    v_coords = np.arange(H, dtype=np.float32)  # filas
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    # Máscara de píxeles válidos (solo la silueta de la persona)
    valid = depth_body > 0

    # Extraer valores válidos
    Z = depth_body[valid]   # profundidad en mm
    u = u_grid[valid]       # columnas de píxeles válidos
    v = v_grid[valid]       # filas de píxeles válidos

    # Proyección inversa pinhole: píxel + profundidad → 3D
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    # Z permanece igual

    # Apilar en matriz (N, 3)
    points = np.column_stack([X, Y, Z]).astype(np.float32)

    # Colores correspondientes normalizados a 0-1
    rgb_valid = rgb_body[valid].astype(np.float32) / 255.0
    colors = rgb_valid

    return points, colors


def create_open3d_pointcloud(points: np.ndarray,
                              colors: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Crea un objeto PointCloud de Open3D a partir de puntos y colores.

    Open3D es la librería que usaremos para visualizar y procesar
    la nube de puntos en 3D.

    Args:
        points: np.ndarray (N, 3) float32 — X, Y, Z en mm
        colors: np.ndarray (N, 3) float32 — RGB normalizado 0-1

    Returns:
        pcd: o3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def remove_outliers(pcd: o3d.geometry.PointCloud,
                    nb_neighbors: int = 20,
                    std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """
    Elimina puntos atípicos (outliers) de la nube de puntos.

    Usa el filtro estadístico de Open3D: para cada punto calcula
    la distancia media a sus N vecinos más cercanos. Los puntos
    cuya distancia supera (media + std_ratio * desviación estándar)
    se consideran outliers y se eliminan.

    Args:
        pcd:          nube de puntos Open3D
        nb_neighbors: número de vecinos para el análisis (default 20)
        std_ratio:    umbral en desviaciones estándar (default 2.0)

    Returns:
        nube de puntos limpia sin outliers
    """
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd_clean


def estimate_normals(pcd: o3d.geometry.PointCloud,
                     radius: float = 50.0,
                     max_nn: int = 30) -> o3d.geometry.PointCloud:
    """
    Estima las normales de la nube de puntos.

    Las normales son necesarias para la reconstrucción de malla (mesh)
    en pasos posteriores. Se calculan usando los vecinos más cercanos
    de cada punto.

    Args:
        pcd:    nube de puntos Open3D
        radius: radio de búsqueda de vecinos en mm (default 50mm)
        max_nn: máximo número de vecinos (default 30)

    Returns:
        nube de puntos con normales estimadas
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    # Orientar normales apuntando hacia la cámara (origen)
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.0])
    )
    return pcd


def reconstruct_pointcloud(depth_body: np.ndarray,
                            rgb_body: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Función principal: reconstrucción 3D completa desde depth + RGB.

    Ejecuta todos los pasos en orden:
        1. Proyección inversa → nube de puntos cruda
        2. Eliminación de outliers
        3. Estimación de normales

    Args:
        depth_body: mapa de profundidad segmentado (H, W) float32 en mm
        rgb_body:   imagen RGB segmentada (H, W, 3) uint8

    Returns:
        pcd: nube de puntos 3D limpia con normales
    """
    # Paso 1 — proyección inversa
    points, colors = depth_to_pointcloud(depth_body, rgb_body)
    print(f"  [1] Nube cruda:       {len(points):,} puntos")

    # Paso 2 — crear objeto Open3D
    pcd = create_open3d_pointcloud(points, colors)

    # Paso 3 — eliminar outliers
    pcd = remove_outliers(pcd)
    print(f"  [2] Sin outliers:     {len(pcd.points):,} puntos")

    # Paso 4 — estimar normales
    pcd = estimate_normals(pcd)
    print(f"  [3] Normales estimadas")

    return pcd