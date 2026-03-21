"""
visualization.py
----------------
Genera el reporte visual final del pipeline en formato PDF.

El reporte contiene 4 secciones en páginas separadas:
    1. Comparación antes/después del preprocesamiento
    2. Silueta de profundidad segmentada con líneas de medición
    3. Nube de puntos 3D renderizada como imagen estática
    4. Tabla de medidas antropométricas

Inputs:
    - frame:        dict con 'rgb' y 'depth' originales
    - depth_clean:  np.ndarray (H, W) float32 preprocesado
    - seg:          dict resultado de segment_body
    - measurements: dict resultado de extract_measurements
    - output_path:  ruta donde guardar el PDF

Outputs:
    - PDF guardado en output_path
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para generar PDF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


# Zonas y sus colores para las líneas de medición
ZONE_COLORS = {
    "cuello":  ("cyan",   287),
    "pecho":   ("yellow", 375),
    "cintura": ("lime",   448),
    "cadera":  ("orange", 487),
}


def render_preprocessing_comparison(depth_raw: np.ndarray,
                                     depth_clean: np.ndarray) -> plt.Figure:
    """
    Página 1 — Comparación antes/después del preprocesamiento.

    Muestra el mapa de profundidad crudo vs el preprocesado
    para visualizar el efecto de la limpieza.

    Args:
        depth_raw:   depth original con inválidos
        depth_clean: depth preprocesado

    Returns:
        figura matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Paso 2 — Preprocesamiento del mapa de profundidad",
                 fontsize=14, fontweight="bold", y=1.02)

    # Antes
    depth_viz_raw = depth_raw.copy().astype(float)
    depth_viz_raw[(depth_raw == 0) | (depth_raw >= 65535)] = np.nan
    axes[0].imshow(depth_viz_raw, cmap="plasma", vmin=300, vmax=4000)
    axes[0].set_title("Antes — con inválidos y ruido", fontsize=12)
    axes[0].axis("off")

    # Después
    im = axes[1].imshow(depth_clean, cmap="plasma", vmin=300, vmax=4000)
    axes[1].set_title("Después — limpio y suavizado", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Profundidad (mm)", shrink=0.8)

    plt.tight_layout()
    return fig


def render_segmentation(depth_clean: np.ndarray,
                         seg: dict,
                         measurements: dict) -> plt.Figure:
    """
    Página 2 — Silueta segmentada con líneas de medición.

    Muestra el depth limpio, la máscara binaria y la silueta
    aislada con las líneas de medición superpuestas.

    Args:
        depth_clean:  depth preprocesado
        seg:          resultado de segment_body
        measurements: resultado de extract_measurements

    Returns:
        figura matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Paso 3 — Segmentación de la silueta corporal",
                 fontsize=14, fontweight="bold", y=1.02)

    # Depth completo
    axes[0].imshow(depth_clean, cmap="plasma", vmin=1000, vmax=4000)
    axes[0].set_title("Depth preprocesado", fontsize=12)
    axes[0].axis("off")

    # Máscara
    axes[1].imshow(seg["mask"], cmap="gray")
    axes[1].set_title("Máscara de silueta", fontsize=12)
    axes[1].axis("off")

    # Silueta con líneas de medición
    depth_body_viz = seg["depth_body"].copy().astype(float)
    depth_body_viz[depth_body_viz == 0] = np.nan
    axes[2].imshow(depth_body_viz, cmap="plasma", vmin=1000, vmax=1700)
    axes[2].set_title("Silueta con zonas de medición", fontsize=12)
    axes[2].axis("off")

    for zone, (color, y) in ZONE_COLORS.items():
        if zone in measurements:
            circ = measurements[zone]["circumference_cm"]
            axes[2].axhline(y=y, color=color, linewidth=2, linestyle="--", alpha=0.9)
            axes[2].text(455, y - 9, f"{zone}: {circ}cm",
                        color=color, fontsize=8, fontweight="bold")

    plt.tight_layout()
    return fig


def render_pointcloud_snapshot(depth_body: np.ndarray) -> plt.Figure:
    """
    Página 3 — Nube de puntos 3D renderizada como imagen estática.

    Proyecta la nube de puntos en una vista frontal coloreada
    por profundidad. No requiere Open3D ni ventana interactiva.

    Args:
        depth_body: depth segmentado (H, W) float32 en mm

    Returns:
        figura matplotlib
    """
    # Parámetros intrínsecos D455
    fx, fy, cx, cy = 674.42, 649.46, 640.0, 360.0

    H, W = depth_body.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth_body > 0

    Z = depth_body[valid].astype(np.float64)
    X = (u[valid] - cx) * Z / fx
    Y = (v[valid] - cy) * Z / fy

    # Colorear por Z
    z_norm = (Z - Z.min()) / (Z.max() - Z.min())
    colors = plt.get_cmap("plasma")(z_norm)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Paso 4 — Nube de puntos 3D",
                 fontsize=14, fontweight="bold", y=1.02)

    # Vista frontal (X, Y)
    ax1 = fig.add_subplot(131)
    ax1.scatter(X, -Y, c=colors, s=0.3, linewidths=0)
    ax1.set_title("Vista frontal", fontsize=11)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # Vista lateral (Z, Y)
    ax2 = fig.add_subplot(132)
    ax2.scatter(Z, -Y, c=colors, s=0.3, linewidths=0)
    ax2.set_title("Vista lateral", fontsize=11)
    ax2.set_aspect("equal")
    ax2.axis("off")

    # Vista superior (X, Z)
    ax3 = fig.add_subplot(133)
    ax3.scatter(X, Z, c=colors, s=0.3, linewidths=0)
    ax3.set_title("Vista superior", fontsize=11)
    ax3.set_aspect("equal")
    ax3.axis("off")

    plt.tight_layout()
    return fig


def render_measurements_table(measurements: dict) -> plt.Figure:
    """
    Página 4 — Tabla de medidas antropométricas.

    Presenta las medidas en una tabla clara con colores
    por zona y un resumen de contornos en cm.

    Args:
        measurements: dict resultado de extract_measurements

    Returns:
        figura matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Paso 5 — Medidas antropométricas",
                 fontsize=14, fontweight="bold", y=1.02)
    ax.axis("off")

    # Datos de la tabla
    table_data = []
    for zone, data in measurements.items():
        table_data.append([
            zone.capitalize(),
            f"{data['width_mm']:.0f} mm",
            f"{data['delta_mm']:.0f} mm",
            f"{data['circumference_mm']:.0f} mm",
            f"{data['circumference_cm']:.1f} cm",
        ])

    col_labels = ["Zona", "Ancho frontal", "Grosor estimado",
                  "Contorno (mm)", "Contorno (cm)"]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.3, 2.5)

    # Estilo encabezados
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C2C2A")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colores por zona
    zone_colors_bg = {
        "cuello":  "#E6F1FB",
        "pecho":   "#FAEEDA",
        "cintura": "#EAF3DE",
        "cadera":  "#EEEDFE",
    }
    for i, zone in enumerate(measurements.keys()):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(
                zone_colors_bg.get(zone, "#F1EFE8")
            )

    # Nota metodológica
    ax.text(0.5, 0.05,
            "Método: proyección inversa pinhole + modelo elíptico + fórmula de Ramanujan\n"
            "Cámara: Intel RealSense D455 | Resolución: 1280×720 | FOV: H=87° V=58°",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    return fig


def generate_report(frame: dict,
                     depth_clean: np.ndarray,
                     seg: dict,
                     measurements: dict,
                     output_path: str | Path = "output/reporte_final.pdf") -> None:
    """
    Función principal: genera el PDF completo con las 4 secciones.

    Args:
        frame:        dict con 'rgb' y 'depth' originales
        depth_clean:  np.ndarray preprocesado
        seg:          resultado de segment_body
        measurements: resultado de extract_measurements
        output_path:  ruta de salida del PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Generando reporte PDF en {output_path}...")

    with PdfPages(str(output_path)) as pdf:

        # Página 1 — Preprocesamiento
        fig1 = render_preprocessing_comparison(frame["depth"], depth_clean)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)
        print("  [1] Página preprocesamiento ✓")

        # Página 2 — Segmentación
        fig2 = render_segmentation(depth_clean, seg, measurements)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)
        print("  [2] Página segmentación ✓")

        # Página 3 — Nube de puntos
        fig3 = render_pointcloud_snapshot(seg["depth_body"])
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)
        print("  [3] Página nube de puntos ✓")

        # Página 4 — Tabla de medidas
        fig4 = render_measurements_table(measurements)
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)
        print("  [4] Página medidas ✓")

        # Metadata del PDF
        d = pdf.infodict()
        d["Title"]   = "Reporte 3D Body Reconstruction"
        d["Author"]  = "Pipeline RealSense D455"
        d["Subject"] = "Medidas antropométricas automatizadas"

    print(f"  ✓ PDF guardado: {output_path}")