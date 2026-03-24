"""
multi_view_main.py
------------------
Pipeline multi-vista usando el código original validado.
"""

import sys, cv2, numpy as np, open3d as o3d
from pathlib import Path
sys.path.insert(0, ".")

from src.multi_view_pipeline import (
    process_single_view, combine_measurements, VIEW_CONFIG
)
from src.multiview_report import generate_full_report


def main():
    data_dir  = Path("data/multiview")
    output_dir = Path("output/multiview")
    (output_dir / "meshes").mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  PIPELINE MULTI-VISTA")
    print("=" * 55)

    view_names = ["frontal", "posterior", "lateral_izq", "lateral_der"]
    view_results = {}
    meshes     = {}
    views_data = []

    for name in view_names:
        print(f"\n[Vista: {name}]")
        rgb   = cv2.cvtColor(
            cv2.imread(str(data_dir / f"{name}_rgb.png")), cv2.COLOR_BGR2RGB
        )
        depth = np.load(str(data_dir / f"{name}_depth.npy")).astype("float32")

        result = process_single_view(rgb, depth, name)
        view_results[name] = result

        # Guardar malla individual
        mesh_path = output_dir / "meshes" / f"{name}.ply"
        o3d.io.write_point_cloud(
            str(output_dir / "meshes" / f"{name}.pcd"),
            result["pcd"]
        )
        print(f"  ✓ Guardada nube: {name}.pcd")

        meshes[name]  = result["pcd"]
        views_data.append({"name": name, "rgb": rgb})

    # Combinar medidas de las 4 vistas
    print("\n[Mediciones combinadas]")
    measurements = combine_measurements(
        view_results["frontal"],
        view_results["posterior"],
        view_results["lateral_izq"],
        view_results["lateral_der"],
    )

    # Altura desde frontal
    height_px = view_results["frontal"]["height_px"]
    depth_ref = float(np.median(
        view_results["frontal"]["depth_body"][
            view_results["frontal"]["depth_body"] > 0
        ]
    ))
    from src.reconstruction import FY, CY
    height_mm = height_px * depth_ref / FY
    height_cm = height_mm / 10.0
    print(f"\n  Altura estimada: {height_cm:.1f} cm")

    # Preparar diag para el reporte
    diag = {}
    for zone, val in measurements.items():
        if val:
            diag[zone] = {
                "w_front_cm": round(val["width_mm"] / 10, 1),
                "w_side_cm":  round(val["depth_mm"] / 10, 1),
                "perim_cm":   val["circ_cm"],
            }
        else:
            diag[zone] = {"w_front_cm": None, "w_side_cm": None, "perim_cm": None}

    meas_simple = {k: (v["circ_cm"] if v else None) for k, v in measurements.items()}

    # Reporte PDF
    print("\n[Generando reporte PDF]")
    generate_full_report(
        views_data   = views_data,
        meshes       = {k: v["pcd"] for k, v in view_results.items()},
        measurements = meas_simple,
        diag         = diag,
        height_cm    = height_cm,
        mesh_ref     = None,
        output_path  = output_dir / "reporte_multiview.pdf",
    )

    # Resumen
    print("\n" + "=" * 55)
    print("  RESUMEN")
    print("=" * 55)
    print(f"  Altura: {height_cm:.1f} cm")
    for k, v in meas_simple.items():
        print(f"  {k:12s}: {f'{v:.1f} cm' if v else 'no disponible'}")
    print("=" * 55)


if __name__ == "__main__":
    main()
