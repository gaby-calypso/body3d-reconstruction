"""
multi_view_main.py — Pipeline multi-vista con mallas individuales por vista.
"""

import argparse, sys
import numpy as np
import open3d as o3d
from pathlib import Path
sys.path.insert(0, ".")

from src.multi_view_loader          import load_all_views
from src.preprocessing              import preprocess_depth
from src.multi_view_reconstruction  import (depth_to_pointcloud_adaptive,
                                            preprocess_pcd, reconstruct_mesh)
from src.per_view_measurements      import extract_measurements_from_views
from src.multiview_report           import generate_full_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default="data/multiview/")
    p.add_argument("--output_dir", type=str, default="output/multiview/")
    p.add_argument("--poisson_depth", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    (out / "meshes").mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  PIPELINE MULTI-VISTA — BODY 3D RECONSTRUCTION")
    print("=" * 55)

    # ── 1. Cargar vistas ───────────────────────────────────────────────────────
    print("\n[1] Cargando vistas...")
    views = load_all_views(data_dir=args.data_dir)

    # ── 2. Reconstruir malla por vista ─────────────────────────────────────────
    print("\n[2] Reconstruyendo malla por vista...")
    meshes     = {}
    views_data = []

    for view in views:
        print(f"\n  Vista: {view['name']}")
        depth_clean = preprocess_depth(view["depth"])
        pcd         = depth_to_pointcloud_adaptive(depth_clean, view["rgb"])
        pcd_clean   = preprocess_pcd(pcd, voxel_size=0.005)
        mesh        = reconstruct_mesh(pcd_clean, depth_param=args.poisson_depth)

        mesh_path = out / "meshes" / f"{view['name']}.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)

        meshes[view["name"]] = mesh
        views_data.append({**view, "mesh": mesh})
        print(f"  ✓ Guardada: {mesh_path}")

    # ── 3. Mediciones por vista ────────────────────────────────────────────────
    print("\n[3] Extrayendo medidas antropométricas...")
    measurements, diag = extract_measurements_from_views(meshes)

    # Altura desde vista frontal
    verts_f    = np.asarray(meshes["frontal"].vertices)
    height_cm  = (verts_f[:,1].max() - verts_f[:,1].min()) * 100

    # ── 4. SMPL (opcional) ─────────────────────────────────────────────────────
    print("\n[4] Ajustando SMPL...")
    mesh_ref = None
    try:
        from src.smpl_fitting import fit_smpl_to_mesh
        smpl_result = fit_smpl_to_mesh(meshes["frontal"])
        mesh_ref    = smpl_result.get("mesh")
        print("  ✓ SMPL ajustado")
    except Exception as e:
        print(f"  ⚠ SMPL no disponible: {e}")

    # ── 5. Reporte PDF ─────────────────────────────────────────────────────────
    print("\n[5] Generando reporte PDF...")
    generate_full_report(
        views_data   = views_data,
        meshes       = meshes,
        measurements = measurements,
        diag         = diag,
        height_cm    = height_cm,
        mesh_ref     = mesh_ref,
        output_path  = out / "reporte_multiview.pdf",
    )

    # ── Resumen ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESUMEN DE MEDIDAS")
    print("=" * 55)
    print(f"  Altura estimada : {height_cm:.1f} cm")
    for k, v in measurements.items():
        print(f"  {k:15s}: {f'{v:.1f} cm' if v else 'no disponible'}")
    print("=" * 55)
    print(f"\n  PDF: {out}/reporte_multiview.pdf")


if __name__ == "__main__":
    main()
