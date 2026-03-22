"""
multiview_report.py
--------------------
Genera reporte PDF completo con:
  Página 1 — 4 vistas RGB originales
  Página 2 — 4 mallas 3D renderizadas (snapshot matplotlib)
  Página 3 — Tabla de medidas con diagrama anatómico
  Página 4 — Comparación de mallas (real vs referencia SMPL si disponible)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import open3d as o3d
from typing import Optional


def _mesh_to_image(mesh: o3d.geometry.TriangleMesh, angle_y: float = 0.0,
                   w: int = 480, h: int = 640) -> np.ndarray:
    """Renderiza malla centrada y normalizada como imagen numpy."""
    verts = np.asarray(mesh.vertices).copy()
    tris  = np.asarray(mesh.triangles)

    # Centrar en X y Z, preservar Y (altura)
    verts[:, 0] -= (verts[:, 0].max() + verts[:, 0].min()) / 2
    verts[:, 2] -= (verts[:, 2].max() + verts[:, 2].min()) / 2

    fig = plt.figure(figsize=(4, 6), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')

    step = max(1, len(tris) // 5000)
    tris_sub = tris[::step]

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = verts[tris_sub]
    col   = Poly3DCollection(polys, alpha=0.75, linewidth=0)

    # Color por altura Y
    y_vals = polys[:, :, 1].mean(axis=1)
    y_n = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-9)
    col.set_facecolor(plt.cm.plasma(y_n))
    ax.add_collection3d(col)

    pad = 0.05
    ax.set_xlim(verts[:,0].min()-pad, verts[:,0].max()+pad)
    ax.set_ylim(verts[:,1].min()-pad, verts[:,1].max()+pad)
    ax.set_zlim(verts[:,2].min()-pad, verts[:,2].max()+pad)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=15, azim=angle_y)

    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    return img


def page_rgb_views(views_data: list) -> plt.Figure:
    """Página 1: 4 imágenes RGB originales."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    fig.suptitle("Vistas RGB de entrada", fontsize=14, fontweight="bold")
    for i, v in enumerate(views_data):
        axes[i].imshow(v["rgb"])
        axes[i].set_title(v["name"], fontsize=11)
        axes[i].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_mesh_views(meshes: dict) -> plt.Figure:
    """Página 2: 4 mallas 3D renderizadas."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    fig.suptitle("Mallas 3D reconstruidas por vista", fontsize=14, fontweight="bold")

    angles = {"frontal": -90, "posterior": 90, "lateral_izq": 0, "lateral_der": 180}
    for i, (name, mesh) in enumerate(meshes.items()):
        angle = angles.get(name, 0)
        img = _mesh_to_image(mesh, angle_y=angle)
        axes[i].imshow(img)
        axes[i].set_title(name, fontsize=11)
        axes[i].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_measurements(measurements: dict, diag: dict, height_cm: float) -> plt.Figure:
    """Página 3: Tabla de medidas + silueta anatómica esquemática."""
    fig, (ax_sil, ax_tab) = plt.subplots(1, 2, figsize=(14, 10),
                                          gridspec_kw={"width_ratios": [1, 1.4]})
    fig.suptitle("Medidas Antropométricas", fontsize=15, fontweight="bold")

    # ── Silueta esquemática ────────────────────────────────────────────────────
    ax_sil.set_xlim(0, 10)
    ax_sil.set_ylim(0, 20)
    ax_sil.set_aspect("equal")
    ax_sil.axis("off")
    ax_sil.set_title("Zonas de medición", fontsize=11)

    # Silueta simplificada (rectángulos apilados)
    body_parts = [
        ("cabeza",   8.5, 19.0, 3.0, 2.0, "#E8D5C4"),
        ("cuello",   8.5, 17.5, 1.2, 1.0, "#F0C8A0"),
        ("hombros",  8.5, 16.2, 5.5, 1.2, "#D4A57A"),
        ("torso",    8.5, 13.5, 4.5, 3.5, "#C8956A"),
        ("cintura",  8.5, 12.0, 3.5, 1.2, "#C8956A"),
        ("cadera",   8.5, 10.5, 5.0, 2.0, "#B87A5A"),
        ("muslos",   8.5,  7.5, 4.5, 3.5, "#A06840"),
        ("rodillas", 8.5,  5.5, 3.5, 1.0, "#906030"),
        ("piernas",  8.5,  3.0, 3.5, 3.0, "#905840"),
        ("pies",     8.5,  1.5, 4.5, 1.2, "#804830"),
    ]
    for _, cx, cy, bw, bh, color in body_parts:
        rect = patches.FancyBboxPatch(
            (cx - bw/2, cy - bh/2), bw, bh,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="#5A3020", linewidth=0.8
        )
        ax_sil.add_patch(rect)

    # Líneas de medición con etiquetas
    measure_y = {
        "cuello":  17.5, "pecho":  15.8, "brazo":   15.0,
        "cintura": 12.0, "cadera": 10.5, "muslo":    7.5, "rodilla": 5.5,
    }
    for zone, y_pos in measure_y.items():
        val = measurements.get(zone)
        color = "#1a7a1a" if val else "#cc3333"
        label = f"{val:.1f} cm" if val else "—"
        ax_sil.annotate(
            "", xy=(11.5, y_pos), xytext=(10.5, y_pos),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.5)
        )
        ax_sil.text(11.6, y_pos, f"{zone}: {label}", fontsize=9,
                    va="center", color=color, fontweight="bold" if val else "normal")

    # ── Tabla de medidas ───────────────────────────────────────────────────────
    ax_tab.axis("off")
    ax_tab.set_title("Resumen de medidas", fontsize=11)

    rows = []
    for zone, val in measurements.items():
        d = diag.get(zone, {})
        wf = f"{d.get('w_front_cm', '--')}" if d.get('w_front_cm') else "--"
        ws = f"{d.get('w_side_cm',  '--')}" if d.get('w_side_cm')  else "--"
        perim = f"{val:.1f} cm" if val else "—"
        rows.append([zone.capitalize(), wf + " cm", ws + " cm", perim])

    headers = ["Zona", "Ancho frontal", "Prof. lateral", "Perímetro"]
    table = ax_tab.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        bbox=[0.0, 0.05, 1.0, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C2C2A")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F1EFE8")
        cell.set_edgecolor("#D3D1C7")

    # Altura estimada
    ax_tab.text(0.5, 0.0, f"Altura estimada: {height_cm:.1f} cm",
                transform=ax_tab.transAxes, ha="center", fontsize=11,
                color="#444", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_mesh_comparison(mesh_real: o3d.geometry.TriangleMesh,
                          mesh_ref:  Optional[o3d.geometry.TriangleMesh] = None) -> plt.Figure:
    """Página 4: Comparación malla real vs referencia."""
    ncols = 2 if mesh_ref is not None else 1
    fig, axes = plt.subplots(1, ncols * 2, figsize=(14, 7))
    fig.suptitle("Comparación de mallas 3D", fontsize=14, fontweight="bold")

    view_angles = [(-90, "frontal"), (0, "lateral")]
    for i, (angle, label) in enumerate(view_angles):
        img_real = _mesh_to_image(mesh_real, angle_y=angle)
        axes[i].imshow(img_real)
        axes[i].set_title(f"Real — {label}", fontsize=10)
        axes[i].axis("off")

        if mesh_ref is not None:
            img_ref = _mesh_to_image(mesh_ref, angle_y=angle)
            axes[i + 2].imshow(img_ref)
            axes[i + 2].set_title(f"Referencia SMPL — {label}", fontsize=10)
            axes[i + 2].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def generate_full_report(
    views_data:   list,
    meshes:       dict,
    measurements: dict,
    diag:         dict,
    height_cm:    float,
    mesh_ref:     Optional[o3d.geometry.TriangleMesh],
    output_path:  str | Path,
) -> None:
    """Genera el PDF completo."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Generando PDF: {output_path}")

    with PdfPages(str(output_path)) as pdf:
        # P1: RGB
        if views_data:
            f1 = page_rgb_views(views_data)
            pdf.savefig(f1, bbox_inches="tight"); plt.close(f1)
            print("  [P1] Vistas RGB ✓")

        # P2: Mallas 3D
        f2 = page_mesh_views(meshes)
        pdf.savefig(f2, bbox_inches="tight"); plt.close(f2)
        print("  [P2] Mallas 3D ✓")

        # P3: Medidas
        f3 = page_measurements(measurements, diag, height_cm)
        pdf.savefig(f3, bbox_inches="tight"); plt.close(f3)
        print("  [P3] Medidas ✓")

        # P4: Comparación
        mesh_frontal = meshes.get("frontal", list(meshes.values())[0])
        f4 = page_mesh_comparison(mesh_frontal, mesh_ref)
        pdf.savefig(f4, bbox_inches="tight"); plt.close(f4)
        print("  [P4] Comparación ✓")

        d = pdf.infodict()
        d["Title"]  = "Reporte Body 3D Reconstruction — Multi-vista"
        d["Author"] = "Pipeline RealSense D455"

    print(f"  ✓ PDF guardado: {output_path}")
