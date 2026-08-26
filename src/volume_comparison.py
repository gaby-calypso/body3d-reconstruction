"""
volume_comparison.py
--------------------
Superpone malla real y sintética, colorea diferencia volumétrica.
    Rojo  → real > sintética (exceso)
    Azul  → real < sintética (déficit)
    Blanco → sin diferencia

Fondo blanco, puntos grandes, 4 vistas: frontal, lateral D, posterior, lateral I.
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from typing import Tuple, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.colorbar


def align_meshes(verts_real: np.ndarray,
                 verts_synth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c_real  = verts_real.mean(axis=0)
    c_synth = verts_synth.mean(axis=0)
    vr = verts_real  - c_real
    vs = verts_synth - c_synth
    height_real  = vr[:, 1].max() - vr[:, 1].min()
    height_synth = vs[:, 1].max() - vs[:, 1].min()
    if height_synth > 1e-6:
        vs = vs * (height_real / height_synth)
    return vr, vs


def compute_vertex_distances(verts_real: np.ndarray,
                              verts_synth: np.ndarray) -> np.ndarray:
    tree = cKDTree(verts_synth)
    distances, idx = tree.query(verts_real)
    r_real  = np.sqrt(verts_real[:, 0]**2  + verts_real[:, 2]**2)
    r_synth = np.sqrt(verts_synth[idx, 0]**2 + verts_synth[idx, 2]**2)
    sign = np.sign(r_real - r_synth)
    sign[sign == 0] = 1.0
    return sign * distances


def distances_to_colors(signed_distances: np.ndarray,
                         percentile_clip: float = 95.0) -> np.ndarray:
    d = signed_distances.copy()
    vmax = np.percentile(np.abs(d), percentile_clip)
    d = np.clip(d, -vmax, vmax)
    d_norm = d / vmax if vmax > 1e-8 else np.zeros_like(d)

    colors = np.ones((len(d_norm), 3))
    pos_mask = d_norm > 0
    neg_mask = d_norm < 0
    colors[pos_mask, 1] = 1.0 - d_norm[pos_mask]
    colors[pos_mask, 2] = 1.0 - d_norm[pos_mask]
    colors[neg_mask, 0] = 1.0 + d_norm[neg_mask]
    colors[neg_mask, 1] = 1.0 + d_norm[neg_mask]
    return np.clip(colors, 0, 1)


def create_comparison_mesh(verts_real: np.ndarray,
                            verts_synth: np.ndarray,
                            faces: np.ndarray
                            ) -> Tuple[o3d.geometry.TriangleMesh,
                                       o3d.geometry.TriangleMesh,
                                       np.ndarray]:
    vr, vs = align_meshes(verts_real, verts_synth)
    signed_dists = compute_vertex_distances(vr, vs)
    colors = distances_to_colors(signed_dists)

    mesh_real = o3d.geometry.TriangleMesh()
    mesh_real.vertices      = o3d.utility.Vector3dVector(vr)
    mesh_real.triangles     = o3d.utility.Vector3iVector(faces)
    mesh_real.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_real.compute_vertex_normals()

    mesh_synth = o3d.geometry.TriangleMesh()
    mesh_synth.vertices  = o3d.utility.Vector3dVector(vs)
    mesh_synth.triangles = o3d.utility.Vector3iVector(faces)
    gray = np.full((len(vs), 3), 0.75)
    mesh_synth.vertex_colors = o3d.utility.Vector3dVector(gray)
    mesh_synth.compute_vertex_normals()

    return mesh_real, mesh_synth, signed_dists


def save_comparison_figure(verts_real: np.ndarray,
                            verts_synth: np.ndarray,
                            signed_dists: np.ndarray,
                            output_path: str = "output/volume_comparison.png"
                            ) -> None:
    """
    Guarda figura con fondo blanco, puntos grandes, 4 vistas.
    """
    vr, vs = align_meshes(verts_real, verts_synth)
    colors = distances_to_colors(signed_dists)

    # 4 vistas + colorbar
    views = [
        ("Frontal",   vr[:, 0],  vr[:, 1], vs[:, 0],  vs[:, 1]),
        ("Lateral D", vr[:, 2],  vr[:, 1], vs[:, 2],  vs[:, 1]),
        ("Posterior", -vr[:, 0], vr[:, 1], -vs[:, 0], vs[:, 1]),
        ("Lateral I", -vr[:, 2], vr[:, 1], -vs[:, 2], vs[:, 1]),
    ]

    fig, axes = plt.subplots(
        1, 5, figsize=(22, 10),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.06]}
    )
    fig.patch.set_facecolor("white")

    for ax, (title, rx, ry, sx, sy) in zip(axes[:4], views):
        ax.set_facecolor("white")

        # Referencia sintética en gris claro detrás
        ax.scatter(sx, sy, c="#CCCCCC", s=2.0,
                   linewidths=0, alpha=0.5, zorder=1)

        # Malla real coloreada — puntos grandes
        ax.scatter(rx, ry, c=colors, s=5.0,
                   linewidths=0, zorder=2)

        ax.set_title(title, fontsize=12, fontweight="bold",
                     color="#2C2C2A", pad=8)
        ax.set_aspect("equal")
        ax.axis("off")

    # Colorbar
    vmax_cm = np.percentile(np.abs(signed_dists), 95) * 100
    norm = matplotlib.colors.Normalize(vmin=-vmax_cm, vmax=vmax_cm)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rdwbu", [(0, "#1565C0"), (0.5, "#FFFFFF"), (1.0, "#C62828")]
    )
    cb = matplotlib.colorbar.ColorbarBase(
        axes[4], cmap=cmap, norm=norm, orientation="vertical"
    )
    cb.set_label("Diferencia (cm)", color="#2C2C2A", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="#2C2C2A")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#2C2C2A", fontsize=9)

    # Estadísticas
    excess_cm  = signed_dists[signed_dists > 0].mean() * 100 \
                 if (signed_dists > 0).any() else 0
    deficit_cm = signed_dists[signed_dists < 0].mean() * 100 \
                 if (signed_dists < 0).any() else 0
    pct_red    = (signed_dists > 0).mean() * 100
    pct_blue   = (signed_dists < 0).mean() * 100

    stats = (
        f"Exceso promedio: +{excess_cm:.1f} cm  |  "
        f"Déficit promedio: {deficit_cm:.1f} cm  |  "
        f"Zonas en exceso: {pct_red:.0f}%  |  "
        f"Zonas en déficit: {pct_blue:.0f}%"
    )
    fig.text(0.5, 0.02, stats, ha="center", va="bottom",
             color="#2C2C2A", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#F5F5F3",
                       edgecolor="#CCCCCC"))

    # Leyenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#CCCCCC', markersize=8,
               label='Referencia sintética'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#C62828', markersize=8,
               label='Exceso de volumen'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#1565C0', markersize=8,
               label='Déficit de volumen'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.06),
               facecolor="white", edgecolor="#CCCCCC")

    fig.suptitle(
        "Comparación volumétrica — Malla real vs Referencia sintética",
        color="#2C2C2A", fontsize=14, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  ✓ Figura guardada en {output_path}")


def compute_zone_volumes(mesh_real: o3d.geometry.TriangleMesh,
                          mesh_synth: o3d.geometry.TriangleMesh
                          ) -> Dict[str, Dict]:
    """
    Calcula el VOLUMEN real (cm³) de cada zona corporal — no una
    distancia promedio como compute_zone_statistics, sino un volumen
    físico de verdad: corta ambas mallas (real e ideal) con dos planos
    horizontales en los límites Y de cada zona, y mide el volumen del
    segmento resultante con el teorema de la divergencia (trimesh, sobre
    una malla cerrada/watertight — tanto la plantilla del reshaper como
    SMPL lo son).

    Args:
        mesh_real, mesh_synth: mallas YA ALINEADAS (mismo origen/escala)
            — usar las que devuelve create_comparison_mesh(), no las
            originales sin alinear.

    Returns:
        dict {zona: {"vol_real_cm3", "vol_ref_cm3", "vol_diff_cm3"}}
    """
    import trimesh

    verts_real  = np.asarray(mesh_real.vertices)
    verts_synth = np.asarray(mesh_synth.vertices)
    faces_real  = np.asarray(mesh_real.triangles)
    faces_synth = np.asarray(mesh_synth.triangles)

    y = verts_real[:, 1]
    y_min, y_max = y.min(), y.max()
    height = y_max - y_min

    zones_frac = {
        "cabeza":  (0.85, 1.00),
        "cuello":  (0.75, 0.85),
        "pecho":   (0.55, 0.75),
        "cintura": (0.40, 0.55),
        "cadera":  (0.25, 0.40),
        "muslos":  (0.10, 0.25),
        "piernas": (0.00, 0.10),
    }

    try:
        tm_real  = trimesh.Trimesh(vertices=verts_real,  faces=faces_real,  process=False)
        tm_synth = trimesh.Trimesh(vertices=verts_synth, faces=faces_synth, process=False)
    except Exception as e:
        print(f"  ⚠ No se pudo construir la malla trimesh para volumen: {e}")
        return {}

    def _slab_volume_m3(tm: "trimesh.Trimesh", y_lo: float, y_hi: float) -> float:
        """Volumen (m³) del segmento de la malla entre dos planos horizontales."""
        try:
            cut = tm.slice_plane(plane_origin=[0, y_lo, 0], plane_normal=[0, 1, 0], cap=True)
            if cut is None or len(cut.vertices) == 0:
                return 0.0
            cut = cut.slice_plane(plane_origin=[0, y_hi, 0], plane_normal=[0, -1, 0], cap=True)
            if cut is None or len(cut.vertices) == 0:
                return 0.0
            if not cut.is_watertight:
                # Respaldo: si el corte no cerró perfectamente (caso raro,
                # geometría degenerada en el borde), usar el volumen del
                # casco convexo como aproximación en vez de fallar.
                cut = cut.convex_hull
            return abs(float(cut.volume))
        except Exception as e:
            print(f"  ⚠ Corte de malla falló para zona en Y=[{y_lo:.3f},{y_hi:.3f}]: {e}")
            return 0.0

    stats = {}
    for zone_name, (frac_lo, frac_hi) in zones_frac.items():
        y_lo = y_min + frac_lo * height
        y_hi = y_min + frac_hi * height
        vol_real_cm3  = _slab_volume_m3(tm_real,  y_lo, y_hi) * 1e6
        vol_synth_cm3 = _slab_volume_m3(tm_synth, y_lo, y_hi) * 1e6
        stats[zone_name] = {
            "vol_real_cm3": vol_real_cm3,
            "vol_ref_cm3":  vol_synth_cm3,
            "vol_diff_cm3": vol_real_cm3 - vol_synth_cm3,
        }
    return stats


def compute_zone_statistics(verts_real: np.ndarray,
                             verts_synth: np.ndarray,
                             signed_dists: np.ndarray
                             ) -> Dict[str, Dict]:
    vr, _ = align_meshes(verts_real, verts_synth)
    y = vr[:, 1]
    y_min, y_max = y.min(), y.max()
    height = y_max - y_min

    zones = {
        "cabeza":  (0.85, 1.00),
        "cuello":  (0.75, 0.85),
        "pecho":   (0.55, 0.75),
        "cintura": (0.40, 0.55),
        "cadera":  (0.25, 0.40),
        "muslos":  (0.10, 0.25),
        "piernas": (0.00, 0.10),
    }

    stats = {}
    for zone_name, (frac_lo, frac_hi) in zones.items():
        y_lo = y_min + frac_lo * height
        y_hi = y_min + frac_hi * height
        mask = (y >= y_lo) & (y < y_hi)
        if mask.sum() < 5:
            continue
        d_zone = signed_dists[mask] * 100
        stats[zone_name] = {
            "mean_cm":     float(d_zone.mean()),
            "std_cm":      float(d_zone.std()),
            "pct_excess":  float((d_zone > 0).mean() * 100),
            "pct_deficit": float((d_zone < 0).mean() * 100),
        }
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Comparación del ESCANEO 3D REAL (malla reconstruida del depth, ver
# src/reconstruction.py y src/multi_view_reconstruction.py) contra una
# malla SMPL de referencia con IMC saludable.
#
# Diferencia clave con create_comparison_mesh/save_comparison_figure de
# arriba: esas funciones asumen que malla real y sintética comparten la
# MISMA topología (ambas son SMPL con los mismos ~6890 vértices en el
# mismo orden) — por eso ahí sí tiene sentido usar el mismo array `faces`
# para las dos. Nuestra malla escaneada tiene una triangulación
# completamente distinta (miles de triángulos irregulares del sensor),
# así que la comparación debe hacerse por vecino-más-cercano en 3D
# (align_meshes + compute_vertex_distances de arriba ya son agnósticas a
# la topología — solo usan arrays de puntos), y el resultado se dibuja
# sobre la topología PROPIA del escaneo, no sobre la de SMPL.
# ═══════════════════════════════════════════════════════════════════════════

def compare_scan_to_ideal(scan_mesh: o3d.geometry.TriangleMesh,
                           verts_ideal: np.ndarray,
                           scan_units: str = "mm",
                           flip_y: bool = True,
                           ) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """
    Colorea la malla ESCANEADA real comparándola contra una malla SMPL
    con peso/IMC saludable. Rojo = exceso de volumen respecto al cuerpo
    saludable, azul = déficit — igual semántica de color que
    create_comparison_mesh, pero sin asumir topología compartida.

    Args:
        scan_mesh:   malla reconstruida del depth (ver reconstruct_body()
                     o la nube fusionada de multi_view_reconstruction)
        verts_ideal: vértices (N,3) de la malla SMPL generada con peso
                     ideal (IMC 22), en metros, Y-up (convención SMPL)
        scan_units:  "mm" si scan_mesh viene directo de reconstruction.py
                     (por vista), "m" si ya viene de la fusión multi-vista
                     (que ya trabaja en metros)
        flip_y:      True si scan_mesh usa convención de imagen (Y crece
                     hacia abajo) y hay que invertirla para que quede
                     Y-up como SMPL. La malla fusionada YA queda Y-up
                     (fuse_pointclouds_icp la deja así), así que para esa
                     debe pasarse flip_y=False.

    Returns:
        (mesh_coloreada, signed_dists) — mesh_coloreada usa los vértices
        y triángulos PROPIOS del escaneo, con vertex_colors nuevos;
        signed_dists en metros, uno por vértice del escaneo (mismo orden).
    """
    verts_scan = np.asarray(scan_mesh.vertices).copy()
    faces_scan = np.asarray(scan_mesh.triangles)

    if scan_units == "mm":
        verts_scan = verts_scan / 1000.0
    if flip_y:
        verts_scan[:, 1] *= -1

    vr, vs = align_meshes(verts_scan, verts_ideal)
    signed_dists = compute_vertex_distances(vr, vs)
    colors = distances_to_colors(signed_dists)

    mesh_colored = o3d.geometry.TriangleMesh()
    mesh_colored.vertices      = o3d.utility.Vector3dVector(vr)
    mesh_colored.triangles     = o3d.utility.Vector3iVector(faces_scan)
    mesh_colored.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_colored.compute_vertex_normals()

    return mesh_colored, signed_dists


def save_scan_vs_ideal_figure(mesh_colored: o3d.geometry.TriangleMesh,
                               verts_ideal: np.ndarray,
                               signed_dists: np.ndarray,
                               output_path: str = "output/scan_vs_ideal.png"
                               ) -> None:
    """
    Guarda una figura de 2 vistas (frontal + lateral) con:
      - la malla IDEAL (IMC saludable) como referencia gris de fondo
      - el ESCANEO real encima, coloreado rojo/azul según exceso/déficit
    """
    verts_scan = np.asarray(mesh_colored.vertices)
    colors     = np.asarray(mesh_colored.vertex_colors)
    c_ideal    = verts_ideal - verts_ideal.mean(axis=0)

    views = [
        ("Frontal", verts_scan[:, 0], verts_scan[:, 1], c_ideal[:, 0], c_ideal[:, 1]),
        ("Lateral", verts_scan[:, 2], verts_scan[:, 1], c_ideal[:, 2], c_ideal[:, 1]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 8),
                              gridspec_kw={"width_ratios": [1, 1, 0.06]})
    fig.patch.set_facecolor("white")

    for ax, (title, rx, ry, sx, sy) in zip(axes[:2], views):
        ax.set_facecolor("white")
        ax.scatter(sx, sy, c="#CCCCCC", s=1.5, linewidths=0, alpha=0.5, zorder=1)
        ax.scatter(rx, ry, c=colors, s=2.0, linewidths=0, zorder=2)
        ax.set_title(title, fontsize=12, fontweight="bold", color="#2C2C2A", pad=8)
        ax.set_aspect("equal")
        ax.axis("off")

    vmax_cm = np.percentile(np.abs(signed_dists), 95) * 100
    norm = matplotlib.colors.Normalize(vmin=-vmax_cm, vmax=vmax_cm)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rdwbu", [(0, "#1565C0"), (0.5, "#FFFFFF"), (1.0, "#C62828")]
    )
    cb = matplotlib.colorbar.ColorbarBase(axes[2], cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("Diferencia (cm)", color="#2C2C2A", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="#2C2C2A")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#2C2C2A", fontsize=9)

    excess_cm  = signed_dists[signed_dists > 0].mean() * 100 if (signed_dists > 0).any() else 0
    deficit_cm = signed_dists[signed_dists < 0].mean() * 100 if (signed_dists < 0).any() else 0
    pct_red    = (signed_dists > 0).mean() * 100
    stats = (f"Exceso promedio: +{excess_cm:.1f} cm  |  "
             f"Déficit promedio: {deficit_cm:.1f} cm  |  "
             f"Superficie en exceso: {pct_red:.0f}%")
    fig.text(0.5, 0.03, stats, ha="center", va="bottom", color="#2C2C2A", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F3", edgecolor="#CCCCCC"))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#CCCCCC',
               markersize=8, label='Referencia IMC saludable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
               markersize=8, label='Exceso de volumen'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0',
               markersize=8, label='Déficit de volumen'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.0),
               facecolor="white", edgecolor="#CCCCCC")

    fig.suptitle("Escaneo 3D real vs. referencia de IMC saludable",
                 color="#2C2C2A", fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Figura guardada en {output_path}")