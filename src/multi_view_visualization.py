"""
multi_view_visualization.py
----------------------------
Genera el reporte PDF del pipeline multi-vista con:
    Página 1 — Las 4 vistas RGB y sus depth maps
    Página 2 — Nube de puntos unificada (4 ángulos)
    Página 3 — Malla 3D final (4 ángulos)
    Página 4 — Comparación de malla SMPL (si disponible)
    Página 5 — Tabla completa de medidas antropométricas
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import open3d as o3d
from typing import Optional


def _render_pcd_snapshot(pcd: o3d.geometry.PointCloud, angle_y: float = 0.0) -> np.ndarray:
    """Renderiza snapshot de la nube de puntos desde un ángulo dado."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(pcd)
    vc = vis.get_view_control()
    vc.rotate(angle_y, 0)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (img * 255).astype(np.uint8)


def render_views_page(views_data: list[dict]) -> plt.Figure:
    """Página 1: 4 vistas RGB + depth."""
    n = len(views_data)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle("Vistas de entrada", fontsize=14, fontweight="bold", y=0.98)

    for i, v in enumerate(views_data):
        ax_rgb = axes[0, i]
        ax_dep = axes[1, i]

        ax_rgb.imshow(v["rgb"])
        ax_rgb.set_title(v["name"], fontsize=10)
        ax_rgb.axis("off")

        depth_vis = v["depth"].copy()
        depth_vis[depth_vis == 0] = np.nan
        ax_dep.imshow(depth_vis, cmap="plasma", aspect="auto")
        ax_dep.set_title("depth", fontsize=9)
        ax_dep.axis("off")

    axes[0, 0].set_ylabel("RGB", fontsize=10)
    axes[1, 0].set_ylabel("Depth", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def render_measurements_table(measurements: dict) -> plt.Figure:
    """Página de tabla de medidas antropométricas."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    fig.suptitle("Medidas Antropométricas", fontsize=16, fontweight="bold")

    rows = []
    for name, val in measurements.items():
        val_str = f"{val:.1f} cm" if val is not None else "—"
        rows.append([name.capitalize(), val_str])

    if rows:
        table = ax.table(
            cellText    = rows,
            colLabels   = ["Zona", "Perímetro"],
            cellLoc     = "center",
            loc         = "center",
            bbox        = [0.1, 0.1, 0.8, 0.85],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2C2C2A")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F1EFE8")
            cell.set_edgecolor("#D3D1C7")

    fig.tight_layout()
    return fig


def generate_multiview_report(
    mesh:         o3d.geometry.TriangleMesh,
    pcd:          o3d.geometry.PointCloud,
    measurements: dict,
    smpl_result:  Optional[dict],
    views_data:   Optional[list[dict]],
    output_path:  str | Path,
) -> None:
    """
    Genera el reporte PDF completo multi-vista.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Generando reporte PDF en {output_path}...")

    with PdfPages(str(output_path)) as pdf:

        # Página 1: Vistas de entrada (solo si se procesaron imágenes)
        if views_data:
            fig1 = render_views_page(views_data)
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close(fig1)
            print("  [1] Página de vistas ✓")

        # Página 2: Tabla de medidas
        fig_table = render_measurements_table(measurements)
        pdf.savefig(fig_table, bbox_inches="tight")
        plt.close(fig_table)
        print("  [2] Tabla de medidas ✓")

        # Página 3: Comparación SMPL (si disponible)
        if smpl_result is not None:
            try:
                from src.visualization import render_smpl_comparison
                fig_smpl = render_smpl_comparison(mesh, smpl_result)
                pdf.savefig(fig_smpl, bbox_inches="tight")
                plt.close(fig_smpl)
                print("  [3] Comparación SMPL ✓")
            except Exception as e:
                print(f"  [3] SMPL no incluido en reporte: {e}")

        d = pdf.infodict()
        d["Title"]  = "Reporte Multi-Vista Body 3D Reconstruction"
        d["Author"] = "Pipeline RealSense D455 — Multi-vista"

    print(f"  ✓ PDF guardado: {output_path}")