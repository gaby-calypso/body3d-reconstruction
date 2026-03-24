"""
rename_views.py
---------------
Muestra cada par RGB+depth y te pide escribir el nombre correcto.
Guarda los archivos renombrados en data/sample/.

Uso:
    python3 rename_views.py
"""

import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("data/sample")

# Archivos a procesar — ajusta si es necesario
PAIRS = [
    {
        "rgb":   "data/sample/pose_image_full_body_09_fr.png",
        "depth": "data/sample/depth_image_full_body_09_fr.npy",
    },
    {
        "rgb":   "data/sample/pose_image_full_body_09_esp.png",
        "depth": "data/sample/depth_image_full_body_09_esp.npy",
    },
    {
        "rgb":   "data/sample/pose_image_full_body_09_izq.png",
        "depth": "data/sample/depth_image_full_body_09_izq.npy",
    },
    {
        "rgb":   "data/sample/pose_image_full_body_09_de.png",
        "depth": "data/sample/depth_image_full_body_09_de.npy",
    },
]

VALID_NAMES = ["frontal", "posterior", "lateral_izq", "lateral_der"]


def depth_to_bgr(depth):
    valid = depth[(depth > 300) & (depth < 4000)]
    d_norm = np.zeros_like(depth, dtype=np.uint8)
    if len(valid) > 0:
        mask = (depth > 300) & (depth < 4000)
        d_min, d_max = valid.min(), valid.max()
        if d_max > d_min:
            d_norm[mask] = ((depth[mask]-d_min)/(d_max-d_min)*255).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)


def show_pair(rgb, depth_bgr, original_name, idx, total):
    """Muestra RGB y depth lado a lado. Cierra con cualquier tecla."""
    H = max(rgb.shape[0], depth_bgr.shape[0])
    canvas = np.zeros((H + 60, rgb.shape[1] + depth_bgr.shape[1], 3), dtype=np.uint8)
    canvas[:rgb.shape[0],        :rgb.shape[1]]                              = rgb
    canvas[:depth_bgr.shape[0],  rgb.shape[1]:rgb.shape[1]+depth_bgr.shape[1]] = depth_bgr

    cv2.putText(canvas, "RGB",   (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(canvas, "Depth", (rgb.shape[1]+10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(canvas, f"Imagen {idx+1}/{total}: {original_name} — cierra la ventana y escribe el nombre",
                (10, H + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    win = f"Imagen {idx+1}/{total}: {original_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1400, 700)
    cv2.imshow(win, canvas)
    cv2.waitKey(0)
    cv2.destroyWindow(win)


def main():
    print("=" * 55)
    print("  Renombrador de vistas")
    print(f"  Nombres válidos: {', '.join(VALID_NAMES)}")
    print("=" * 55)

    saved = []

    for idx, pair in enumerate(PAIRS):
        rgb_path   = Path(pair["rgb"])
        depth_path = Path(pair["depth"])

        print(f"\n── Par {idx+1}/{len(PAIRS)}: {rgb_path.name} ──")

        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            print(f"  ERROR: no se encontró {rgb_path}")
            continue

        try:
            depth = np.load(str(depth_path)).astype("float32")
        except Exception as e:
            print(f"  ERROR: no se encontró {depth_path}: {e}")
            continue

        depth_bgr = depth_to_bgr(depth)

        # Mostrar las imágenes
        show_pair(rgb, depth_bgr, rgb_path.stem, idx, len(PAIRS))

        # Pedir nombre en terminal
        while True:
            name = input(f"  Escribe el nombre para esta vista {VALID_NAMES}: ").strip().lower()
            if name in VALID_NAMES:
                break
            elif name == "":
                print("  Saltando esta imagen.")
                name = None
                break
            else:
                print(f"  Nombre no válido. Opciones: {', '.join(VALID_NAMES)}")

        if name is None:
            continue

        # Guardar con nombre correcto
        out_rgb   = OUTPUT_DIR / f"{name}_rgb.png"
        out_depth = OUTPUT_DIR / f"{name}_depth.npy"

        cv2.imwrite(str(out_rgb), rgb)
        np.save(str(out_depth), depth)

        saved.append(name)
        print(f"  ✓ Guardado como: {out_rgb.name}  +  {out_depth.name}")

    # Resumen
    print(f"\n{'='*55}")
    print(f"  Guardadas: {len(saved)}/{len(PAIRS)} vistas")
    for name in saved:
        print(f"    ✓ {name}_rgb.png  +  {name}_depth.npy")
    missing = [v for v in VALID_NAMES if v not in saved]
    if missing:
        print(f"  Faltantes: {', '.join(missing)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()