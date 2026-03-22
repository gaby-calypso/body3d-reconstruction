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
