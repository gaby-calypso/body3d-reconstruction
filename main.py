"""
main.py
-------
Entry point for the 3D body silhouette reconstruction pipeline.

Usage:
    python main.py

Each step of the pipeline is called from here in sequence.
Future steps are commented out and will be activated incrementally.
"""

from pathlib import Path
from src.loader import load_frame


# ── Configuration ─────────────────────────────────────────────────────────────
# Change these paths to point to your actual data files when using the camera.
RGB_PATH   = Path("data/sample/rgb.png")
DEPTH_PATH = Path("data/sample/depth.npy")


def main() -> None:
    print("=" * 50)
    print("  3D Body Reconstruction Pipeline")
    print("=" * 50)

    # ── Step 1: Data Loading ───────────────────────────────────────────────────
    print("\n[Step 1] Loading input data...")
    frame = load_frame(RGB_PATH, DEPTH_PATH)

    rgb   = frame["rgb"]
    depth = frame["depth"]

    print(f"  ✓ RGB image loaded:  shape={rgb.shape},   dtype={rgb.dtype}")
    print(f"  ✓ Depth map loaded:  shape={depth.shape}, dtype={depth.dtype}")
    print(f"  ✓ Depth range:       min={depth.min():.1f} mm, "
          f"max={depth.max():.1f} mm")

    # ── Step 2: Preprocessing ──────────────────────────────────
    print("\n[Step 2] Preprocesando mapa de profundidad...")
    from src.preprocessing import preprocess_depth
    depth_clean = preprocess_depth(depth)
    print(f"✓ Depth preprocesado: shape={depth_clean.shape}, "
          f"dtype={depth_clean.dtype}")

    # ── Step 3: Segmentation (coming later) ───────────────────────────────────
    print("\n[Step 3] Segmentando silueta corporal...")
    from src.segmentation import segment_body
    seg = segment_body(rgb, depth_clean)

    print(f"  ✓ Máscara generada:  shape={seg['mask'].shape}, "
          f"dtype={seg['mask'].dtype}")
    print(f"  ✓ Píxeles del cuerpo: {seg['body_pixels']:,}")

    # ── Step 4: 3D Reconstruction ─────────────────────────────────────────────
    print("\n[Step 4] Reconstruyendo nube de puntos 3D...")
    from src.reconstruction import reconstruct_pointcloud
    pcd = reconstruct_pointcloud(seg["depth_body"], seg["rgb_body"])
    print(f"  ✓ Nube de puntos lista: {len(pcd.points):,} puntos 3D")

    # ── Step 5: Measurements ──────────────────────────────────────────────────
    print("\n[Step 5] Extrayendo medidas antropométricas...")
    from src.measurements import extract_measurements
    measurements = extract_measurements(seg["mask"], seg["depth_body"])

    print("\n  Resumen de medidas:")
    print("  " + "-" * 40)
    for zone, data in measurements.items():
        print(f"  {zone:10s}: {data['circumference_cm']} cm")
    print("  " + "-" * 40)

    # ── Step 6: Visualization — PDF Report ────────────────────────────────────
    print("\n[Step 6] Generando reporte PDF...")
    from src.visualization import generate_report
    generate_report(
        frame       = {"rgb": rgb, "depth": depth},
        depth_clean = depth_clean,
        seg         = seg,
        measurements= measurements,
        output_path = "output/reporte_final.pdf"
    )
    print("  ✓ Reporte generado en output/reporte_final.pdf")

    # ── Step 7: Morphing 3D ───────────────────────────────────────────────────
    print("\n[Step 7] Aplicando transformación 3D...")
    from src.reconstruction import depth_to_pointcloud
    from src.morphing import morph_pointcloud

    # Obtener nube de puntos original
    points_original, _ = depth_to_pointcloud(
        seg["depth_body"], seg["rgb_body"]
    )

    # Definir contornos objetivo — modifica estos valores a tu gusto
    targets = {
        "cintura": 55.0,   # cm objetivo para cintura
        "cadera":  90.0,   # cm objetivo para cadera
    }

    points_morphed, new_measurements = morph_pointcloud(
        points_original, measurements, targets
    )
    print(f"  ✓ Morfing aplicado: {len(points_morphed):,} puntos transformados")

    # Comparar medidas antes y después
    print("\n  Comparación de medidas:")
    print("  " + "-" * 50)
    print(f"  {'Zona':10s} {'Antes':>10s} {'Después':>10s} {'Cambio':>10s}")
    print("  " + "-" * 50)
    for zone in measurements:
        before = measurements[zone]["circumference_cm"]
        after  = new_measurements[zone]["circumference_cm"]
        diff   = after - before
        sign   = "+" if diff >= 0 else ""
        print(f"  {zone:10s} {before:>8.1f}cm {after:>8.1f}cm "
              f"{sign}{diff:>7.1f}cm")
    print("  " + "-" * 50)

    # ── 5 ejemplos de morfing ─────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.morphing import morph_pointcloud

    ejemplos = [
        ("disminucion_cuello",   {"cuello":  25.0}),
        ("disminucion_pecho",    {"pecho":   50.0}),
        ("disminucion_cintura",  {"cintura": 35.0}),
        ("disminucion_cadera",   {"cadera":  50.0}),
        ("cambio_todas",         {"cuello": 25.0, "pecho": 50.0,
                                  "cintura": 35.0, "cadera": 50.0}),
    ]

    for nombre, targets in ejemplos:
        print(f"\n  Ejemplo: {nombre}")
        pts_m, _ = morph_pointcloud(points_original, measurements, targets)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Morfing — {nombre.replace('_', ' ')}",
                     fontsize=13, fontweight="bold")

        # Colorear por Y (altura) — no cambia con la transformación
        Y_orig = points_original[:, 1]
        y_norm = (Y_orig - Y_orig.min()) / (Y_orig.max() - Y_orig.min())
        colors = plt.get_cmap("RdYlBu")(y_norm)

        for ax, pts, title in [
            (axes[0], points_original, "Original"),
            (axes[1], pts_m,           "Transformado"),
        ]:
            ax.scatter(pts[:, 0], -pts[:, 1],
                       c=colors, s=0.5, linewidths=0)
            ax.set_title(title, fontsize=11)
            ax.set_aspect("equal")
            ax.axis("off")

        # Líneas de referencia por zona transformada
        for zone in targets:
            from src.morphing import ZONE_Y_WORLD
            y_w = -ZONE_Y_WORLD[zone]
            for ax in axes:
                ax.axhline(y=y_w, color="white", linewidth=0.8,
                           linestyle="--", alpha=0.6)

        info = "  |  ".join([f"{z}: {v}cm" for z, v in targets.items()])
        fig.text(0.5, 0.01, f"Objetivo → {info}",
                 ha="center", fontsize=9, color="gray", style="italic")

        path = f"output/morfing_{nombre}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Guardado: {path}")

    print("\n  ✓ Todos los ejemplos generados en output/")

    # ── Step 8: SMPL — malla de la persona + variaciones ─────────────────────
    print("\n[Step 8] Generando malla SMPL de la persona y variaciones...")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.smpl_fitting import (load_smpl_model, get_vertices,
                                   get_all_measurements)

    smpl_model = load_smpl_model("models", "neutral")

    # ── Anclar al cuello validado (32.5cm) ────────────────────────────────────
    # beta[0]=-4.3 produce cuello~33cm según calibración
    betas_persona = np.array([-4.3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    meas_persona  = get_all_measurements(smpl_model, betas_persona)

    print("\n  Malla base anclada al cuello (32.5cm validado):")
    print("  " + "-" * 35)
    for z, v in meas_persona.items():
        print(f"    {z:10s}: {v:.1f} cm")
    print("  " + "-" * 35)

    # ── Variaciones relativas desde la malla base ─────────────────────────────
    # Usamos delta sobre beta[0] para cambios de volumen global
    # y combinaciones para cambios más específicos
    variaciones = [
        ("base",          betas_persona.copy(),                    "Base\n(persona)"),
        ("+5cm cintura",  betas_persona + [+0.3,0,0,0,0,0,0,0,0,0], "+volumen\nleve"),
        ("+10cm cintura", betas_persona + [+0.7,0,0,0,0,0,0,0,0,0], "+volumen\nmoderado"),
        ("+20cm cintura", betas_persona + [+1.5,0,0,0,0,0,0,0,0,0], "+volumen\ngrande"),
        ("-5cm cintura",  betas_persona + [-0.5,0,0,0,0,0,0,0,0,0], "-volumen\nleve"),
    ]

    fig, axes = plt.subplots(1, len(variaciones), figsize=(18, 8))
    fig.suptitle("SMPL — variaciones desde malla base de la persona",
                 fontsize=13, fontweight="bold")

    def plot_smpl(ax, verts, title, meas):
        y_norm = (verts[:, 1] - verts[:, 1].min())
        y_norm /= (y_norm.max() + 1e-8)
        colors = plt.get_cmap("RdYlBu_r")(y_norm)
        ax.scatter(verts[:, 0], verts[:, 1],
                   c=colors, s=0.5, linewidths=0)
        ax.set_title(title, fontsize=9)
        # Añadir medidas debajo
        info = (f"cuello:{meas['cuello']:.0f}\n"
                f"pecho:{meas['pecho']:.0f}\n"
                f"cin:{meas['cintura']:.0f}\n"
                f"cad:{meas['cadera']:.0f}")
        ax.text(0.5, -0.05, info, transform=ax.transAxes,
                ha='center', va='top', fontsize=7,
                color='gray', family='monospace')
        ax.set_aspect("equal")
        ax.axis("off")

    print("\n  Medidas por variación:")
    print("  " + "-" * 60)
    print(f"  {'Variación':18s} {'cuello':>8s} {'pecho':>8s} "
          f"{'cintura':>8s} {'cadera':>8s}")
    print("  " + "-" * 60)

    for i, (nombre, betas, titulo) in enumerate(variaciones):
        verts = get_vertices(smpl_model, betas)
        meas  = get_all_measurements(smpl_model, betas)
        plot_smpl(axes[i], verts, titulo, meas)
        print(f"  {nombre:18s} "
              f"{meas['cuello']:>7.1f}cm "
              f"{meas['pecho']:>7.1f}cm "
              f"{meas['cintura']:>7.1f}cm "
              f"{meas['cadera']:>7.1f}cm")

    print("  " + "-" * 60)

    plt.tight_layout()
    plt.savefig("output/smpl_variaciones_persona.png",
                dpi=150, bbox_inches="tight")
    print("\n  ✓ Guardado en output/smpl_variaciones_persona.png")

    # ── Superposición con diferencia de volumen ───────────────────────────────
    print("\n  Generando visualización de diferencias de volumen...")

    def plot_volume_diff(ax, verts_base, verts_mod, title):
        """
        Superpone dos mallas y colorea la diferencia de volumen.
        Rojo = ganó volumen | Azul = perdió volumen | Gris = sin cambio
        """
        # Centrar ambas mallas en el mismo punto (alineación perfecta)
        center_base = np.array([verts_base[:, 0].mean(),
                                  verts_base[:, 1].mean()])
        center_mod  = np.array([verts_mod[:, 0].mean(),
                                  verts_mod[:, 1].mean()])

        vb = verts_base.copy()
        vm = verts_mod.copy()
        vb[:, :2] -= center_base
        vm[:, :2] -= center_mod

        # Filtrar solo el torso (excluir brazos y piernas)
        # Torso: Y entre -0.3 y +0.5, X entre -0.2 y +0.2
        torso_mask_b = (
            (vb[:, 1] >= -0.35) & (vb[:, 1] <= 0.50) &
            (np.abs(vb[:, 0]) <= 0.22)
        )
        torso_mask_m = (
            (vm[:, 1] >= -0.35) & (vm[:, 1] <= 0.50) &
            (np.abs(vm[:, 0]) <= 0.22)
        )

        vb_t = vb[torso_mask_b]
        vm_t = vm[torso_mask_m]

        # Calcular diferencia de ancho (X) punto a punto
        # Usamos los mismos índices para comparar vértice a vértice
        common_idx = np.where(torso_mask_b & torso_mask_m)[0]
        if len(common_idx) == 0:
            return

        diff_x = (np.abs(vm[common_idx, 0]) -
                  np.abs(vb[common_idx, 0]))

        max_diff = np.abs(diff_x).max()
        if max_diff < 1e-6:
            max_diff = 1e-6

        # Colorear por diferencia — colores intensos
        colors = np.zeros((len(common_idx), 4))
        for j, d in enumerate(diff_x):
            if d > 0.001:
                intensity = min(d / max_diff, 1.0)
                colors[j] = [1, 1-intensity, 1-intensity, 1.0]
            elif d < -0.001:
                intensity = min(-d / max_diff, 1.0)
                colors[j] = [1-intensity, 1-intensity, 1, 1.0]
            else:
                colors[j] = [0.6, 0.6, 0.6, 0.8]

        # Malla base en gris oscuro (referencia más visible)
        ax.scatter(vb[common_idx, 0], vb[common_idx, 1],
                   c='gray', s=1.0, linewidths=0, alpha=0.5, zorder=1)

        # Malla modificada coloreada — puntos más grandes
        ax.scatter(vm[common_idx, 0], vm[common_idx, 1],
                   c=colors, s=2.5, linewidths=0, zorder=2)

        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.40, 0.55)
        ax.set_aspect("equal")
        ax.axis("off")

    # Generar comparaciones
    comparaciones = [
        ("+volumen leve",    variaciones[1][1], "+0.3 beta"),
        ("+volumen moderado",variaciones[2][1], "+0.7 beta"),
        ("+volumen grande",  variaciones[3][1], "+1.5 beta"),
        ("-volumen leve",    variaciones[4][1], "-0.5 beta"),
    ]

    verts_base_smpl = get_vertices(smpl_model, betas_persona)

    fig2, axes2 = plt.subplots(1, len(comparaciones), figsize=(16, 8))
    fig2.suptitle(
        "Diferencia de volumen — Rojo: ganado | Azul: perdido | Gris: sin cambio",
        fontsize=12, fontweight="bold"
    )

    for i, (nombre, betas_mod, label) in enumerate(comparaciones):
        verts_mod = get_vertices(smpl_model, betas_mod)
        plot_volume_diff(axes2[i], verts_base_smpl, verts_mod,
                         f"{nombre}\n({label})")

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red',      alpha=0.7, label='Volumen ganado'),
        Patch(facecolor='blue',     alpha=0.7, label='Volumen perdido'),
        Patch(facecolor='lightgray',alpha=0.5, label='Sin cambio'),
    ]
    fig2.legend(handles=legend_elements, loc='lower center',
                ncol=3, fontsize=10, framealpha=0.8,
                bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("output/smpl_diff_volumen.png", dpi=150, bbox_inches="tight")
    print("  ✓ Guardado en output/smpl_diff_volumen.png")

if __name__ == "__main__":
    main()