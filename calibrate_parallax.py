"""
calibrate_parallax.py
---------------------
Herramienta interactiva para medir el offset (paralaje) entre
la imagen RGB y el mapa de profundidad de la D455.

Analiza las 4 vistas una por una para verificar que el paralaje es consistente.

Uso:
    python3 calibrate_parallax.py

Controles:
    Slider X  → desplazamiento horizontal del depth
    Slider Y  → desplazamiento vertical del depth
    Slider A  → transparencia del depth
    N         → siguiente vista
    P         → vista anterior
    Q / ESC   → cerrar e imprimir resumen de todos los valores
"""

import cv2
import numpy as np

# ── Vistas a analizar ─────────────────────────────────────────────────────────
VIEWS = [
    {"name": "frontal",      "rgb": "data/sample/frontal_rgb.png",      "depth": "data/sample/frontal_depth.npy"},
    {"name": "posterior",    "rgb": "data/sample/posterior_rgb.png",    "depth": "data/sample/posterior_depth.npy"},
    {"name": "lateral_izq",  "rgb": "data/sample/lateral_izq_rgb.png",  "depth": "data/sample/lateral_izq_depth.npy"},
    {"name": "lateral_der",  "rgb": "data/sample/lateral_der_rgb.png",  "depth": "data/sample/lateral_der_depth.npy"},
]

RANGE         = 200
SLIDER_CENTER = RANGE
WINDOW        = "Calibracion paralaje  |  N=siguiente  P=anterior  Q=salir"


def depth_to_bgr(depth: np.ndarray) -> np.ndarray:
    valid = depth[(depth > 300) & (depth < 4000)]
    d_norm = np.zeros_like(depth, dtype=np.uint8)
    if len(valid) > 0:
        mask = (depth > 300) & (depth < 4000)
        d_min, d_max = valid.min(), valid.max()
        if d_max > d_min:
            d_norm[mask] = (
                (depth[mask] - d_min) / (d_max - d_min) * 255
            ).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)


def overlay(rgb, depth_bgr, ox, oy, alpha, view_name, view_idx, total):
    H, W = rgb.shape[:2]
    result = rgb.copy()

    src_x1 = max(0, -ox);  src_y1 = max(0, -oy)
    src_x2 = min(W, W-ox); src_y2 = min(H, H-oy)
    dst_x1 = max(0,  ox);  dst_y1 = max(0,  oy)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        region_d = depth_bgr[src_y1:src_y2, src_x1:src_x2]
        region_r = result[dst_y1:dst_y2, dst_x1:dst_x2]
        has_data = region_d.sum(axis=2) > 0
        mask     = has_data.astype(np.float32)[:, :, np.newaxis]
        blended  = alpha * region_d.astype(np.float32) + (1-alpha) * region_r.astype(np.float32)
        result[dst_y1:dst_y2, dst_x1:dst_x2] = (
            mask * blended + (1-mask) * region_r
        ).astype(np.uint8)

    # Cruceta de referencia
    cv2.line(result, (W//2, 0), (W//2, H), (0,255,0), 1)
    cv2.line(result, (0, H//2), (W, H//2), (0,255,0), 1)

    # Info en pantalla
    cv2.putText(result,
        f"Vista {view_idx+1}/{total}: {view_name}   offset_x={ox:+d}   offset_y={oy:+d}   alpha={alpha:.2f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(result,
        "Ajusta sliders hasta alinear bordes del cuerpo.  N=siguiente  P=anterior  Q=terminar",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)

    return result


def load_view(view: dict):
    rgb = cv2.imread(view["rgb"])
    if rgb is None:
        print(f"  ERROR: no se encontro {view['rgb']}")
        return None, None
    try:
        depth = np.load(view["depth"]).astype("float32")
    except Exception as e:
        print(f"  ERROR: no se encontro {view['depth']}: {e}")
        return rgb, None
    return rgb, depth


def main():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 800)

    cv2.createTrackbar("Offset X", WINDOW, SLIDER_CENTER, RANGE*2, lambda v: None)
    cv2.createTrackbar("Offset Y", WINDOW, SLIDER_CENTER, RANGE*2, lambda v: None)
    cv2.createTrackbar("Alpha %",  WINDOW, 50,            100,      lambda v: None)

    # Guardar offset por vista
    offsets = {v["name"]: (0, 0) for v in VIEWS}

    idx = 0
    rgb, depth = load_view(VIEWS[idx])

    print("\nControles: N=siguiente vista  P=anterior  Q=terminar y ver resumen\n")

    while True:
        if rgb is None or depth is None:
            # Vista con error — mostrar pantalla negra con mensaje
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, f"No se pudo cargar: {VIEWS[idx]['name']}",
                        (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            ox    = cv2.getTrackbarPos("Offset X", WINDOW) - SLIDER_CENTER
            oy    = cv2.getTrackbarPos("Offset Y", WINDOW) - SLIDER_CENTER
            alpha = cv2.getTrackbarPos("Alpha %",  WINDOW) / 100.0
            depth_bgr = depth_to_bgr(depth)
            frame = overlay(rgb, depth_bgr, ox, oy, alpha,
                            VIEWS[idx]["name"], idx, len(VIEWS))

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            # Guardar offset de la vista actual antes de salir
            if rgb is not None:
                ox = cv2.getTrackbarPos("Offset X", WINDOW) - SLIDER_CENTER
                oy = cv2.getTrackbarPos("Offset Y", WINDOW) - SLIDER_CENTER
                offsets[VIEWS[idx]["name"]] = (ox, oy)
            break

        elif key in (ord("n"), ord("N")):
            # Guardar offset actual y pasar a la siguiente vista
            if rgb is not None:
                ox = cv2.getTrackbarPos("Offset X", WINDOW) - SLIDER_CENTER
                oy = cv2.getTrackbarPos("Offset Y", WINDOW) - SLIDER_CENTER
                offsets[VIEWS[idx]["name"]] = (ox, oy)
                print(f"  {VIEWS[idx]['name']:15s}: offset_x={ox:+d}  offset_y={oy:+d}")

            idx = (idx + 1) % len(VIEWS)
            rgb, depth = load_view(VIEWS[idx])

            # Restaurar sliders al offset guardado de la nueva vista
            saved_ox, saved_oy = offsets[VIEWS[idx]["name"]]
            cv2.setTrackbarPos("Offset X", WINDOW, SLIDER_CENTER + saved_ox)
            cv2.setTrackbarPos("Offset Y", WINDOW, SLIDER_CENTER + saved_oy)

        elif key in (ord("p"), ord("P")):
            # Guardar offset actual y volver a la vista anterior
            if rgb is not None:
                ox = cv2.getTrackbarPos("Offset X", WINDOW) - SLIDER_CENTER
                oy = cv2.getTrackbarPos("Offset Y", WINDOW) - SLIDER_CENTER
                offsets[VIEWS[idx]["name"]] = (ox, oy)
                print(f"  {VIEWS[idx]['name']:15s}: offset_x={ox:+d}  offset_y={oy:+d}")

            idx = (idx - 1) % len(VIEWS)
            rgb, depth = load_view(VIEWS[idx])

            saved_ox, saved_oy = offsets[VIEWS[idx]["name"]]
            cv2.setTrackbarPos("Offset X", WINDOW, SLIDER_CENTER + saved_ox)
            cv2.setTrackbarPos("Offset Y", WINDOW, SLIDER_CENTER + saved_oy)

    cv2.destroyAllWindows()

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print("  Resumen de paralaje por vista:")
    print(f"{'='*52}")
    xs = [v[0] for v in offsets.values()]
    ys = [v[1] for v in offsets.values()]
    for name, (ox, oy) in offsets.items():
        print(f"  {name:15s}: offset_x={ox:+d}  offset_y={oy:+d}")
    print(f"{'='*52}")
    print(f"  Promedio:        offset_x={int(round(sum(xs)/len(xs))):+d}  offset_y={int(round(sum(ys)/len(ys))):+d}")
    print(f"{'='*52}")
    print("\nCopia el promedio (o el valor de frontal) en pose_overlay.py:")
    print(f"  PARALLAX_X = {int(round(sum(xs)/len(xs)))}")
    print(f"  PARALLAX_Y = {int(round(sum(ys)/len(ys)))}")


if __name__ == "__main__":
    main()