"""
multiview_report.py — Reporte PDF completo multi-vista.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import open3d as o3d
from typing import Optional


def _pcd_to_image(pcd, angle_y: float = 0.0) -> np.ndarray:
    """Renderiza una nube de puntos como imagen matplotlib."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return np.ones((400, 300, 3), dtype=np.uint8) * 240

    # Centrar en XZ
    pts = pts.copy()
    pts[:, 0] -= (pts[:, 0].max() + pts[:, 0].min()) / 2
    pts[:, 2] -= (pts[:, 2].max() + pts[:, 2].min()) / 2

    fig = plt.figure(figsize=(3, 5), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')

    step = max(1, len(pts) // 4000)
    p = pts[::step]

    # Color por altura Y
    y_n = (p[:, 1] - p[:, 1].min()) / (p[:, 1].max() - p[:, 1].min() + 1e-9)
    ax.scatter(p[:, 0], p[:, 2], p[:, 1],
               c=plt.cm.plasma(y_n), s=0.8, alpha=0.7, depthshade=False)

    ax.set_xlim(pts[:,0].min(), pts[:,0].max())
    ax.set_ylim(pts[:,2].min(), pts[:,2].max())
    ax.set_zlim(pts[:,1].min(), pts[:,1].max())
    ax.set_xlabel('X', fontsize=6); ax.set_ylabel('Z', fontsize=6)
    ax.set_zlabel('Y (altura)', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=10, azim=angle_y)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img


def page_rgb_views(views_data: list) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Vistas RGB de entrada", fontsize=14, fontweight="bold")
    for i, v in enumerate(views_data):
        axes[i].imshow(v["rgb"])
        axes[i].set_title(v["name"], fontsize=11)
        axes[i].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_pcd_views(pcds: dict) -> plt.Figure:
    """Página 2: 4 nubes de puntos renderizadas."""
    angles = {"frontal": -90, "posterior": 90, "lateral_izq": 0, "lateral_der": 180}
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    fig.suptitle("Nubes de puntos 3D por vista", fontsize=14, fontweight="bold")
    for i, (name, pcd) in enumerate(pcds.items()):
        angle = angles.get(name, 0)
        img = _pcd_to_image(pcd, angle_y=angle)
        axes[i].imshow(img)
        axes[i].set_title(name, fontsize=11)
        axes[i].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_measurements(measurements: dict, diag: dict, height_cm: float) -> plt.Figure:
    fig, (ax_sil, ax_tab) = plt.subplots(1, 2, figsize=(14, 10),
                                          gridspec_kw={"width_ratios": [1, 1.5]})
    fig.suptitle("Medidas Antropométricas", fontsize=15, fontweight="bold")

    # Silueta esquemática
    ax_sil.set_xlim(0, 14); ax_sil.set_ylim(0, 20)
    ax_sil.set_aspect("equal"); ax_sil.axis("off")
    ax_sil.set_title("Zonas de medición", fontsize=11)

    body_parts = [
        (7.0, 19.0, 2.8, 1.8, "#E8D5C4"),
        (7.0, 17.5, 1.2, 1.0, "#F0C8A0"),
        (7.0, 16.2, 5.5, 1.2, "#D4A57A"),
        (7.0, 13.8, 4.5, 3.5, "#C8956A"),
        (7.0, 12.2, 3.5, 1.2, "#C8956A"),
        (7.0, 10.5, 5.2, 2.0, "#B87A5A"),
        (7.0,  7.5, 4.8, 3.5, "#A06840"),
        (7.0,  5.5, 3.8, 1.0, "#906030"),
        (7.0,  3.0, 3.8, 3.0, "#905840"),
        (7.0,  1.5, 4.8, 1.2, "#804830"),
    ]
    for cx, cy, bw, bh, color in body_parts:
        rect = patches.FancyBboxPatch(
            (cx - bw/2, cy - bh/2), bw, bh,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#5A3020", linewidth=0.8
        )
        ax_sil.add_patch(rect)

    measure_y = {
        "cuello": 17.5, "pecho": 15.8, "brazo": 15.0,
        "cintura": 12.2, "cadera": 10.5, "muslo": 7.5, "rodilla": 5.5,
    }
    for zone, y_pos in measure_y.items():
        val = measurements.get(zone)
        color = "#1a7a1a" if val else "#cc3333"
        label = f"{val:.1f} cm" if val else "—"
        ax_sil.annotate("", xy=(10.5, y_pos), xytext=(9.8, y_pos),
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.5))
        ax_sil.text(10.7, y_pos, f"{zone}: {label}", fontsize=9,
                    va="center", color=color,
                    fontweight="bold" if val else "normal")

    # Tabla
    ax_tab.axis("off")
    ax_tab.set_title("Resumen de medidas", fontsize=11)
    rows = []
    for zone, val in measurements.items():
        d = diag.get(zone, {})
        wf = f"{d['w_front_cm']:.1f}" if d.get('w_front_cm') else "--"
        ws = f"{d['w_side_cm']:.1f}"  if d.get('w_side_cm')  else "--"
        p  = f"{val:.1f} cm"          if val                  else "—"
        rows.append([zone.capitalize(), wf + " cm", ws + " cm", p])

    table = ax_tab.table(
        cellText=rows,
        colLabels=["Zona", "Ancho frontal", "Prof. lateral", "Perímetro"],
        cellLoc="center", loc="center",
        bbox=[0.0, 0.05, 1.0, 0.88],
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

    ax_tab.text(0.5, 0.0, f"Altura estimada: {height_cm:.1f} cm",
                transform=ax_tab.transAxes, ha="center",
                fontsize=11, color="#444", style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_pcd_comparison(pcds: dict) -> plt.Figure:
    """Página 4: las 4 nubes en una vista conjunta frontal y lateral."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Nubes de puntos 3D — vista frontal y lateral", fontsize=14, fontweight="bold")

    for i, (angle, label) in enumerate([(-90, "Vista frontal"), (0, "Vista lateral")]):
        pcd = pcds.get("frontal") if i == 0 else pcds.get("lateral_izq")
        if pcd is None:
            pcd = list(pcds.values())[0]
        img = _pcd_to_image(pcd, angle_y=angle)
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=11)
        axes[i].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def generate_full_report(
    views_data:   list,
    meshes:       dict,
    measurements: dict,
    diag:         dict,
    height_cm:    float,
    mesh_ref,
    output_path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Generando PDF: {output_path}")

    with PdfPages(str(output_path)) as pdf:
        if views_data:
            f1 = page_rgb_views(views_data)
            pdf.savefig(f1, bbox_inches="tight"); plt.close(f1)
            print("  [P1] Vistas RGB ✓")

        f2 = page_pcd_views(meshes)
        pdf.savefig(f2, bbox_inches="tight"); plt.close(f2)
        print("  [P2] Nubes 3D ✓")

        f3 = page_measurements(measurements, diag, height_cm)
        pdf.savefig(f3, bbox_inches="tight"); plt.close(f3)
        print("  [P3] Medidas ✓")

        f4 = page_pcd_comparison(meshes)
        pdf.savefig(f4, bbox_inches="tight"); plt.close(f4)
        print("  [P4] Comparación ✓")

        d = pdf.infodict()
        d["Title"]  = "Reporte Body 3D — Multi-vista"
        d["Author"] = "Pipeline RealSense D455"

    print(f"  ✓ PDF guardado: {output_path}")
