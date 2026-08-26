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


def build_silhouette_mesh(depth_body: np.ndarray,
                           rgb_body: np.ndarray,
                           fx: float = FX,
                           fy: float = FY,
                           cx: float = CX,
                           cy: float = CY,
                           max_edge_jump_mm: float = 25.0,
                           smooth_iterations: int = 5) -> o3d.geometry.TriangleMesh:
    """
    Reconstruye una malla 3D que sigue EXACTAMENTE la silueta segmentada,
    triangulando directamente la grilla del mapa de profundidad.

    Por qué esto es más fiel que "nube de puntos + reconstrucción Poisson":
    Poisson NO conoce la silueta real — imagina una superficie cerrada que
    mejor explica un conjunto de puntos sueltos, y puede "inflar" o
    redondear el contorno más allá del cuerpo real. Aquí, en cambio, ya
    conocemos la conectividad exacta: dos píxeles vecinos del sensor
    (arriba/abajo/izquierda/derecha) representan superficie continua real
    SALVO que el salto de profundidad entre ellos sea grande — en ese caso
    no son la misma superficie (es el borde del cuerpo, o dos partes del
    cuerpo separadas en profundidad, como un brazo despegado del torso) y
    NO se conectan. El resultado es una malla cuyo contorno es, por
    construcción, exactamente el contorno de la máscara segmentada — ni
    más ancha ni más redondeada de lo que el sensor realmente vio.

    Funciona igual para una imagen capturada en vivo que para una
    importada: solo depende del arreglo depth_body ya segmentado.

    Args:
        depth_body:        depth segmentado (H, W) float32 en mm (0 = fondo)
        rgb_body:           RGB segmentado (H, W, 3) uint8
        fx, fy, cx, cy:     intrínsecos de cámara (sin modificar — se usan
                            tal cual se reciben, igual que antes)
        max_edge_jump_mm:   salto máximo de profundidad (mm) entre dos
                            píxeles vecinos para considerarlos parte de la
                            misma superficie continua. Si el salto es mayor,
                            no se traza el triángulo entre ellos (preserva
                            bordes reales en vez de "tapar" el hueco).
        smooth_iterations:  iteraciones de suavizado Taubin — reduce el
                            ruido del sensor sin encoger ni inflar el
                            volumen (a diferencia del suavizado Laplaciano
                            simple, que sí deforma la silueta).

    Returns:
        mesh: o3d.geometry.TriangleMesh con vértices, colores y normales
    """
    H, W = depth_body.shape
    depth_body = depth_body.astype(np.float32)

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    valid = depth_body > 0

    # Proyección inversa pinhole (idéntica a depth_to_pointcloud, sin tocar
    # fx/fy/cx/cy — solo se usan tal cual se reciben)
    X = (uu - cx) * depth_body / fx
    Y = (vv - cy) * depth_body / fy
    Z = depth_body

    # Índice de vértice por píxel (-1 = píxel inválido / fondo)
    vertex_index = np.full((H, W), -1, dtype=np.int64)
    vertex_index[valid] = np.arange(int(valid.sum()))

    points = np.column_stack([X[valid], Y[valid], Z[valid]]).astype(np.float64)
    colors = (rgb_body[valid].astype(np.float64) / 255.0)

    # Bloques 2x2 de la grilla → 2 triángulos cada uno
    v00 = vertex_index[:-1, :-1]
    v01 = vertex_index[:-1, 1:]
    v10 = vertex_index[1:, :-1]
    v11 = vertex_index[1:, 1:]

    z00, z01 = Z[:-1, :-1], Z[:-1, 1:]
    z10, z11 = Z[1:, :-1], Z[1:, 1:]

    def _continuous(za, zb):
        return np.abs(za.astype(np.float32) - zb.astype(np.float32)) < max_edge_jump_mm

    # Triángulo A: (v00, v10, v01) — mitad superior-izquierda del bloque
    okA = (v00 >= 0) & (v10 >= 0) & (v01 >= 0) \
        & _continuous(z00, z10) & _continuous(z00, z01) & _continuous(z10, z01)
    # Triángulo B: (v01, v10, v11) — mitad inferior-derecha del bloque
    okB = (v01 >= 0) & (v10 >= 0) & (v11 >= 0) \
        & _continuous(z01, z10) & _continuous(z01, z11) & _continuous(z10, z11)

    tris_a = np.column_stack([v00[okA], v10[okA], v01[okA]])
    tris_b = np.column_stack([v01[okB], v10[okB], v11[okB]])
    triangles = np.concatenate([tris_a, tris_b], axis=0)

    mesh = o3d.geometry.TriangleMesh()
    if len(points) == 0 or len(triangles) == 0:
        return mesh  # silueta vacía — devuelve malla vacía en vez de fallar

    mesh.vertices      = o3d.utility.Vector3dVector(points)
    mesh.triangles     = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    if smooth_iterations > 0 and len(mesh.triangles) > 0:
        # Taubin preserva el volumen/silueta — no la "encoge" como Laplace
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)

    mesh.compute_vertex_normals()
    return mesh


def mesh_to_pointcloud(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
    """Extrae una nube de puntos (con normales) desde una malla ya construida."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        pcd.normals = mesh.vertex_normals
    return pcd


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


def reconstruct_body(depth_body: np.ndarray,
                      rgb_body: np.ndarray,
                      fx: float = FX,
                      fy: float = FY,
                      cx: float = CX,
                      cy: float = CY) -> dict:
    """
    Función principal: reconstrucción 3D fiel a la silueta desde depth + RGB.

    A diferencia de la versión anterior (nube de puntos no estructurada +
    filtro estadístico genérico + normales por vecinos-más-cercanos), esto
    triangula directamente la grilla del depth respetando los saltos de
    profundidad reales, así que la silueta resultante es la que el sensor
    realmente capturó — ni redondeada ni "inflada" por la reconstrucción.

    Funciona igual para una vista capturada en vivo que para una importada:
    solo depende del arreglo depth_body/rgb_body ya segmentado.

    Args:
        depth_body: mapa de profundidad segmentado (H, W) float32 en mm
        rgb_body:   imagen RGB segmentada (H, W, 3) uint8
        fx,fy,cx,cy: intrínsecos de cámara (se usan tal cual, sin tocar)

    Returns:
        dict con:
            "mesh" -> o3d.geometry.TriangleMesh de la silueta exacta
            "pcd"  -> o3d.geometry.PointCloud derivada de la misma malla
                      (mismos vértices/colores/normales — no una extracción
                      independiente, para que ambos sean siempre consistentes)
    """
    # Paso 1 — malla exacta de la silueta (triangulación de la grilla depth)
    mesh = build_silhouette_mesh(depth_body, rgb_body, fx, fy, cx, cy)
    print(f"  [1] Malla silueta:    {len(mesh.vertices):,} vértices, "
          f"{len(mesh.triangles):,} triángulos")

    # Paso 2 — limpieza residual muy leve (picos puntuales del sensor que
    # el umbral de salto de profundidad no alcanzó a filtrar). Se aplica
    # sobre la nube derivada de la malla, no sobre una extracción aparte.
    pcd = mesh_to_pointcloud(mesh)
    if len(pcd.points) > 0:
        pcd, keep_idx = pcd.remove_statistical_outlier(nb_neighbors=12, std_ratio=3.0)
        print(f"  [2] Limpieza fina:    {len(pcd.points):,} puntos "
              f"({len(keep_idx):,}/{len(mesh.vertices):,} conservados)")

    return {"mesh": mesh, "pcd": pcd}


def reconstruct_pointcloud(depth_body: np.ndarray,
                            rgb_body: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Reconstrucción 3D completa desde depth + RGB (compatibilidad).

    Mantiene la misma firma y tipo de retorno que siempre — cualquier
    código existente que llame a esta función sigue funcionando igual,
    solo que ahora internamente usa la reconstrucción fiel a la silueta
    (ver reconstruct_body / build_silhouette_mesh) en vez de la nube de
    puntos no estructurada + filtro genérico anterior.

    Args:
        depth_body: mapa de profundidad segmentado (H, W) float32 en mm
        rgb_body:   imagen RGB segmentada (H, W, 3) uint8

    Returns:
        pcd: nube de puntos 3D limpia con normales
    """
    return reconstruct_body(depth_body, rgb_body)["pcd"]