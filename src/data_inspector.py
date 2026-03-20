"""
inspect.py
----------
Inspección visual rápida de los datos cargados.
Muestra la imagen RGB y el mapa de profundidad lado a lado.

Uso:
    python3 src/inspect.py
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

RGB_PATH   = Path("data/sample/rgb.png")
DEPTH_PATH = Path("data/sample/depth.npy")

# Cargar datos
rgb   = cv2.cvtColor(cv2.imread(str(RGB_PATH)), cv2.COLOR_BGR2RGB)
depth = np.load(str(DEPTH_PATH)).astype(np.float32)

# Estadísticas del depth
valid_mask  = (depth > 0) & (depth < 65535)
valid_depth = depth[valid_mask]

print(f"Píxeles totales:       {depth.size:,}")
print(f"Píxeles válidos:       {valid_mask.sum():,} "
      f"({100 * valid_mask.mean():.1f}%)")
print(f"Píxeles inválidos:     {(~valid_mask).sum():,} "
      f"({100 * (1 - valid_mask.mean()):.1f}%)")
print(f"Profundidad válida:    min={valid_depth.min():.0f} mm, "
      f"max={valid_depth.max():.0f} mm")
print(f"Profundidad promedio:  {valid_depth.mean():.0f} mm")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Inspección de datos — RealSense D455", fontsize=13)

# RGB
axes[0].imshow(rgb)
axes[0].set_title("Imagen RGB")
axes[0].axis("off")

# Depth — ocultamos los inválidos para visualizar mejor
depth_viz = depth.copy()
depth_viz[~valid_mask] = np.nan

im = axes[1].imshow(depth_viz, cmap="plasma", vmin=valid_depth.min(),
                    vmax=valid_depth.max())
axes[1].set_title("Mapa de profundidad (mm)")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], label="mm")

plt.tight_layout()
plt.savefig("output/inspeccion_datos.png", dpi=150, bbox_inches="tight")
print("\n✓ Imagen guardada en output/inspeccion_datos.png")
plt.show()

# ── Visualización comparativa: antes vs después del preprocesamiento ──────────
import sys
sys.path.insert(0, ".")
from src.preprocessing import preprocess_depth

print("\nAplicando preprocesamiento para comparación visual...")
depth_clean = preprocess_depth(depth)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Preprocesamiento — antes vs después", fontsize=13)

# Antes
depth_viz_raw = depth.copy()
depth_viz_raw[(depth == 0) | (depth >= 65535)] = np.nan
axes2[0].imshow(depth_viz_raw, cmap="plasma", vmin=300, vmax=4000)
axes2[0].set_title(f"Antes — {(~valid_mask).sum():,} huecos")
axes2[0].axis("off")

# Después
im2 = axes2[1].imshow(depth_clean, cmap="plasma", vmin=300, vmax=4000)
axes2[1].set_title("Después — limpio y suavizado")
axes2[1].axis("off")
plt.colorbar(im2, ax=axes2[1], label="mm")

plt.tight_layout()
plt.savefig("output/comparacion_preprocesamiento.png", dpi=150, bbox_inches="tight")
print("✓ Guardado en output/comparacion_preprocesamiento.png")
plt.show()