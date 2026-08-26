from __future__ import annotations
import numpy as np
import open3d as o3d
import cv2

DEFAULT_INTRINSICS = {
    "fx": 637.8, "fy": 637.8,
    "cx": 640.5, "cy": 360.5,
    "width": 1280, "height": 720,
}

def depth_to_pointcloud_adaptive(
    depth: np.ndarray,
    rgb:   np.ndarray,
    intrinsics: dict = DEFAULT_INTRINSICS,
    depth_min_mm: float = 300.0,
    depth_max_mm: float = 4000.0,
    roi: dict = None,
) -> o3d.geometry.PointCloud:
    """Genera nube de puntos desde depth con ROI espacial + rango de profundidad."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    h, w = depth.shape

    # ROI por defecto calibrada con la herramienta de sliders
    if roi is None:
        roi = {
            "x1": int(w * 0.31), "x2": int(w * 0.59),
            "y1": 0,             "y2": int(h * 0.87),
            "d_min": 1410,       "d_max": 1790,
        }

    x1, x2 = roi["x1"], roi["x2"]
    y1, y2 = roi["y1"], roi["y2"]
    d_min  = roi["d_min"]
    d_max  = roi["d_max"]

    # Máscara: ROI espacial + rango de profundidad
    mask = np.zeros((h, w), dtype=np.uint8)
    roi_depth = depth[y1:y2, x1:x2]
    roi_mask  = ((roi_depth >= d_min) & (roi_depth <= d_max)).astype(np.uint8) * 255

    # Limpieza morfológica
    kernel = np.ones((5, 5), np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN,  kernel)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

    # Componente conectada más grande
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        roi_mask = ((labels == largest) * 255).astype(np.uint8)

    mask[y1:y2, x1:x2] = roi_mask

    # Convertir a nube de puntos
    ys, xs = np.where(mask == 255)
    z = depth[ys, xs] / 1000.0
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]

    X = (xs - cx) * z / fx
    Y = -(ys - cy) * z / fy
    Z = z

    points = np.stack([X, Y, Z], axis=1)
    colors = rgb[ys, xs].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"    Nube local: {len(pcd.points):,} puntos  depth=[{d_min}, {d_max}] mm  ROI x=[{x1},{x2}] y=[{y1},{y2}]")
    return pcd

def depth_to_pointcloud(depth, mask, rgb, intrinsics=DEFAULT_INTRINSICS):
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    ys, xs = np.where(mask == 255)
    z = depth[ys, xs] / 1000.0
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    X = (xs - cx) * z / fx
    Y = -(ys - cy) * z / fy
    Z = z
    points = np.stack([X, Y, Z], axis=1)
    colors = rgb[ys, xs].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def rotate_pointcloud(pcd, angle_deg, flip_x=False):
    pts  = np.asarray(pcd.points).copy()
    cols = np.asarray(pcd.colors).copy()
    if flip_x:
        pts[:, 0] *= -1
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    x_new =  cos_a * pts[:, 0] + sin_a * pts[:, 2]
    z_new = -sin_a * pts[:, 0] + cos_a * pts[:, 2]
    pts[:, 0] = x_new
    pts[:, 2] = z_new
    pcd_rot = o3d.geometry.PointCloud()
    pcd_rot.points = o3d.utility.Vector3dVector(pts)
    pcd_rot.colors = o3d.utility.Vector3dVector(cols)
    return pcd_rot

def preprocess_pcd(pcd, voxel_size=0.005):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd_clean.orient_normals_consistent_tangent_plane(30)
    return pcd_clean

def fuse_pointclouds(pcds):
    combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined += pcd
    combined_clean, _ = combined.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
    return combined_clean

def reconstruct_mesh(pcd, depth_param=9):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
    pcd.orient_normals_consistent_tangent_plane(50)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth_param)
    densities_np = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities_np < np.percentile(densities_np, 10))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    print(f"  Malla Poisson: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangulos")
    return mesh

def center_pointcloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Centra la nube en XZ (no en Y para preservar altura)."""
    pts = np.asarray(pcd.points).copy()
    cols = np.asarray(pcd.colors).copy()
    # Centrar solo X y Z, no Y (altura debe preservarse)
    pts[:, 0] -= np.median(pts[:, 0])
    pts[:, 2] -= np.median(pts[:, 2])
    pcd_c = o3d.geometry.PointCloud()
    pcd_c.points = o3d.utility.Vector3dVector(pts)
    pcd_c.colors = o3d.utility.Vector3dVector(cols)
    return pcd_c


def register_icp(source: o3d.geometry.PointCloud,
                  target: o3d.geometry.PointCloud,
                  max_corr_dist: float = 0.03,
                  init: np.ndarray = None):
    """
    Refina la alineación de 'source' contra 'target' con ICP punto-a-plano.

    La rotación fija por ángulo asumido (rotate_pointcloud) es solo un
    punto de partida — asume que la persona no se movió ni un milímetro
    entre tomas y que el ángulo de cámara es exactamente el nominal. ICP
    corrige esa desalineación residual real, minimizando la distancia
    entre la superficie de 'source' y la de 'target' ya fusionada.

    Args:
        source, target: nubes de puntos en METROS, con normales estimadas
        max_corr_dist:  distancia máxima (m) para considerar dos puntos
                        correspondientes — debe ser del orden del error
                        esperado de la alineación inicial (unos cm)
        init:           transformación inicial 4x4 (por defecto identidad,
                        ya que rotate_pointcloud dejó la nube en la zona
                        correcta aproximada)

    Returns:
        (transformation 4x4, fitness [0-1], inlier_rmse en metros)
    """
    if init is None:
        init = np.eye(4)
    if not source.has_normals():
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max_corr_dist, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max_corr_dist, max_nn=30))

    reg = o3d.pipelines.registration.registration_icp(
        source, target, max_corr_dist, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    return reg.transformation, reg.fitness, reg.inlier_rmse


# Ángulos de partida asumidos por vista (grados, rotación sobre eje Y).
# Son una ALINEACIÓN INICIAL para que ICP tenga de dónde arrancar — no
# necesitan ser exactos, pero si están muy alejados del ángulo real de
# captura, ICP puede converger a un mínimo local incorrecto. Si al
# fusionar ves partes del cuerpo duplicadas o mal orientadas, ajusta
# estos valores según tu protocolo real de captura (¿la persona gira en
# sentido horario o antihorario? ¿"lateral_der" es el lado derecho de
# la persona o el lado derecho visto por la cámara?).
DEFAULT_VIEW_ANGLES = {
    "frontal":     {"angle":    0, "flip_x": False},
    "lateral_der": {"angle":  -90, "flip_x": False},
    "posterior":   {"angle":  180, "flip_x": False},
    "lateral_izq": {"angle":   90, "flip_x": False},
}
_FUSION_ORDER = ["frontal", "lateral_der", "posterior", "lateral_izq"]


def fuse_pointclouds_icp(pcds_mm: dict,
                          view_angles: dict = None,
                          voxel_size: float = 0.006,
                          icp_max_corr_dist: float = 0.03) -> dict:
    """
    Fusiona en un solo cuerpo 3D las nubes de puntos por vista que produce
    src.reconstruction.reconstruct_body() (en milímetros, eje Y creciendo
    hacia abajo — convención de imagen).

    Para cada vista, en orden frontal → lateral_der → posterior →
    lateral_izq:
        1. Convierte a metros e invierte Y (Y-up, misma convención que
           el resto de este módulo y que SMPL).
        2. Centra en XZ y rota al ángulo asumido de esa vista (punto de
           partida, ver DEFAULT_VIEW_ANGLES).
        3. Si ya hay una nube fusionada previa, refina la alineación de
           esta vista contra ella con ICP punto-a-plano (register_icp) —
           esto es lo que corrige la posición real, no solo la asumida.
        4. Suma la vista alineada a la nube fusionada.

    Args:
        pcds_mm:     dict {view_name: o3d.geometry.PointCloud} en mm,
                     tal como quedan en STATE.pcds
        view_angles: override de DEFAULT_VIEW_ANGLES si tu protocolo de
                     captura usa otros ángulos/orden
        voxel_size:  tamaño de voxel (m) para downsample — también define
                     la escala de detalle que sobrevive a la fusión
        icp_max_corr_dist: distancia máxima (m) para emparejar puntos
                     en ICP — subirlo ayuda si la alineación inicial es
                     peor de lo esperado, pero puede converger mal si es
                     demasiado grande

    Returns:
        dict con:
            "fused_pcd":       nube de puntos fusionada y limpia (metros)
            "per_view_fitness": {view_name: fitness ICP 0-1} — valores
                bajos (<0.3) sugieren que esa vista no se alineó bien
                (revisar el ángulo asumido para esa vista)
    """
    view_angles = view_angles or DEFAULT_VIEW_ANGLES
    names = [n for n in _FUSION_ORDER
             if n in pcds_mm and pcds_mm[n] is not None and len(pcds_mm[n].points) > 0]

    if not names:
        return {"fused_pcd": o3d.geometry.PointCloud(), "per_view_fitness": {}}

    fused = None
    per_view_fitness = {}

    for name in names:
        src = pcds_mm[name]
        pts = np.asarray(src.points).copy() / 1000.0     # mm -> m
        pts[:, 1] *= -1                                    # Y-down -> Y-up
        cols = (np.asarray(src.colors).copy()
                if src.has_colors() else np.ones_like(pts) * 0.6)

        pcd_m = o3d.geometry.PointCloud()
        pcd_m.points = o3d.utility.Vector3dVector(pts)
        pcd_m.colors = o3d.utility.Vector3dVector(cols)

        cfg = view_angles.get(name, {"angle": 0, "flip_x": False})
        pcd_centered = center_pointcloud(pcd_m)
        pcd_rot      = rotate_pointcloud(pcd_centered, cfg["angle"], cfg.get("flip_x", False))
        pcd_clean    = preprocess_pcd(pcd_rot, voxel_size)

        if fused is None:
            fused = pcd_clean
            per_view_fitness[name] = 1.0
            print(f"  [fusión] {name:12s}: ancla — {len(pcd_clean.points):,} puntos")
        else:
            transform, fitness, rmse = register_icp(
                pcd_clean, fused, max_corr_dist=icp_max_corr_dist
            )
            pcd_clean.transform(transform)
            per_view_fitness[name] = float(fitness)
            print(f"  [fusión] {name:12s}: ICP fitness={fitness:.2f}  "
                  f"rmse={rmse*1000:.1f}mm  → {len(pcd_clean.points):,} puntos")
            if fitness < 0.3:
                print(f"    ⚠ fitness bajo — revisar el ángulo asumido para '{name}' "
                      f"en DEFAULT_VIEW_ANGLES")
            fused = fused + pcd_clean
            fused = fused.voxel_down_sample(voxel_size)

    fused_clean, _ = fused.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
    fused_clean.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30))
    fused_clean.orient_normals_consistent_tangent_plane(30)
    print(f"  [fusión] total: {len(fused_clean.points):,} puntos "
          f"(de {len(names)} vista(s))")

    return {"fused_pcd": fused_clean, "per_view_fitness": per_view_fitness}


def reconstruct_from_views(views_data, voxel_size=0.005, poisson_depth=9):
    rotated_pcds = []
    for v in views_data:
        # 1. Centrar la nube local en XZ antes de rotar
        pcd_centered = center_pointcloud(v["pcd"])
        # 2. Rotar al ángulo de vista
        pcd_rot   = rotate_pointcloud(pcd_centered, v["angle"], v.get("flip_x", False))
        # 3. Limpiar
        pcd_clean = preprocess_pcd(pcd_rot, voxel_size)
        rotated_pcds.append(pcd_clean)
        pts = np.asarray(pcd_clean.points)
        print(f"  ok {v['name']:15s}: {len(pcd_clean.points):,} puntos  "
              f"X=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]  "
              f"Z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
    pcd_unified = fuse_pointclouds(rotated_pcds)
    print(f"  Nube unificada: {len(pcd_unified.points):,} puntos totales")
    mesh = reconstruct_mesh(pcd_unified, depth_param=poisson_depth)
    return pcd_unified, mesh