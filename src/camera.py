"""
camera.py
---------
Módulo de captura con Intel RealSense D455.

Modos:
    REAL      → usa pyrealsense2 con la cámara física conectada
    SIMULATED → genera frames sintéticos para desarrollo sin cámara

Uso básico:
    cam = RealSenseCamera(simulate=True)
    cam.start()
    rgb, depth = cam.get_frames()
    cam.stop()

Para usar con cámara real:
    cam = RealSenseCamera(simulate=False)
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CameraConfig:
    """Configuración de la cámara RealSense D455."""
    width:      int   = 640
    height:     int   = 480
    fps:        int   = 30
    depth_min:  float = 0.3    # metros — distancia mínima válida
    depth_max:  float = 3.0    # metros — distancia máxima válida


class RealSenseCamera:
    """
    Interfaz unificada para la Intel RealSense D455.

    En modo simulación genera:
        - RGB  : imagen sintética de figura humana con gradiente
        - Depth: mapa de profundidad sintético con forma elipsoidal

    En modo real usa pyrealsense2 con alineación depth→color
    para que ambos frames estén en el mismo sistema de coordenadas.

    Args:
        simulate : True para modo simulación, False para cámara real
        config   : configuración de resolución y FPS
    """

    def __init__(self,
                 simulate: bool = True,
                 config: CameraConfig = CameraConfig()):
        self.simulate  = simulate
        self.config    = config
        self._running  = False
        self._pipeline = None
        self._align    = None
        self._frame_n  = 0        # contador para animación simulada

    # ── API pública ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Inicia la cámara o el generador de simulación."""
        if self.simulate:
            self._running = True
            print("[Camera] Modo SIMULACIÓN activo")
            print(f"         Resolución: {self.config.width}x{self.config.height}"
                  f" @ {self.config.fps}fps")
        else:
            self._start_realsense()

    def stop(self) -> None:
        """Detiene la cámara y libera recursos."""
        self._running = False
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        print("[Camera] Detenida")

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene un par de frames (RGB, Depth) sincronizados.

        Returns:
            rgb   : (H, W, 3) uint8  — imagen BGR (compatible con OpenCV)
            depth : (H, W)    float32 — profundidad en metros
                    valores 0.0 indican píxeles sin dato válido

        Raises:
            RuntimeError si la cámara no está iniciada
        """
        if not self._running:
            raise RuntimeError("Cámara no iniciada. Llama a start() primero.")

        if self.simulate:
            return self._generate_simulated_frames()
        else:
            return self._get_realsense_frames()

    def get_depth_colormap(self, depth: np.ndarray) -> np.ndarray:
        """
        Convierte el mapa de profundidad a imagen coloreada (JET)
        para visualización en la GUI.

        Args:
            depth : (H, W) float32 en metros

        Returns:
            (H, W, 3) uint8 imagen BGR con colormap JET
        """
        # Normalizar al rango [depth_min, depth_max]
        d_clip = np.clip(depth, self.config.depth_min, self.config.depth_max)
        d_norm = ((d_clip - self.config.depth_min) /
                  (self.config.depth_max - self.config.depth_min) * 255
                  ).astype(np.uint8)
        return cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Modo REAL — RealSense D455 ────────────────────────────────────────────

    def _start_realsense(self) -> None:
        """
        Configura e inicia el pipeline RealSense D455.

        Streams habilitados:
            - Color : 640×480 BGR8  30fps
            - Depth : 640×480 Z16   30fps

        Filtros aplicados:
            - Decimation  : reduce ruido espacial
            - Spatial     : suavizado preservando bordes
            - Temporal    : estabilidad temporal
            - Hole filling: rellena huecos sin dato
        """
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 no instalado.\n"
                "Instala con: pip3 install pyrealsense2\n"
                "O usa simulate=True para desarrollo sin cámara."
            )

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.color,
            self.config.width, self.config.height,
            rs.format.bgr8, self.config.fps
        )
        cfg.enable_stream(
            rs.stream.depth,
            self.config.width, self.config.height,
            rs.format.z16, self.config.fps
        )

        profile = self._pipeline.start(cfg)

        # Alineación depth → color (mismo sistema de coordenadas)
        self._align = rs.align(rs.stream.color)

        # Filtros de postprocesado para mejorar calidad del depth
        self._filter_dec   = rs.decimation_filter()
        self._filter_dec.set_option(rs.option.filter_magnitude, 1)

        self._filter_spat  = rs.spatial_filter()
        self._filter_spat.set_option(rs.option.filter_magnitude,    2)
        self._filter_spat.set_option(rs.option.filter_smooth_alpha, 0.5)
        self._filter_spat.set_option(rs.option.filter_smooth_delta, 20)

        self._filter_temp  = rs.temporal_filter()
        self._filter_temp.set_option(rs.option.filter_smooth_alpha, 0.4)
        self._filter_temp.set_option(rs.option.filter_smooth_delta, 20)

        self._filter_hole  = rs.hole_filling_filter()

        # Obtener escala de profundidad (convierte raw → metros)
        depth_sensor   = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        self._running = True
        print("[Camera] RealSense D455 iniciada")
        print(f"         Depth scale: {self._depth_scale:.5f} m/unit")
        print(f"         Resolución:  {self.config.width}x{self.config.height}"
              f" @ {self.config.fps}fps")

    def _get_realsense_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lee y procesa un par de frames de la cámara real."""
        import pyrealsense2 as rs

        # Esperar frames alineados
        frames        = self._pipeline.wait_for_frames(timeout_ms=5000)
        aligned       = self._align.process(frames)
        color_frame   = aligned.get_color_frame()
        depth_frame   = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Frame inválido recibido de la cámara")

        # Aplicar filtros al depth
        depth_frame = self._filter_spat.process(depth_frame)
        depth_frame = self._filter_temp.process(depth_frame)
        depth_frame = self._filter_hole.process(depth_frame)

        # Convertir a numpy
        rgb   = np.asanyarray(color_frame.get_data())         # (H,W,3) uint8
        depth_raw = np.asanyarray(depth_frame.get_data())     # (H,W)   uint16

        # Convertir a metros y enmascarar valores fuera de rango
        depth = depth_raw.astype(np.float32) * self._depth_scale
        depth[(depth < self.config.depth_min) |
              (depth > self.config.depth_max)] = 0.0

        return rgb, depth

    # ── Modo SIMULACIÓN ───────────────────────────────────────────────────────

    def _generate_simulated_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera frames sintéticos realistas para pruebas sin cámara.

        RGB  : fondo gris + silueta humana con textura de piel
        Depth: elipsoide 3D que simula un cuerpo a ~1.5m de distancia
        """
        self._frame_n += 1
        h, w = self.config.height, self.config.width
        cx, cy = w // 2, h // 2

        # ── RGB sintético ─────────────────────────────────────────────────────
        rgb = np.full((h, w, 3), 80, dtype=np.uint8)   # fondo gris oscuro

        # Silueta humana simplificada: cabeza + torso + piernas
        body_parts = [
            # (cx, cy, radio_x, radio_y, color_BGR)
            (cx, cy - 150, 40,  45,  (120, 160, 200)),   # cabeza
            (cx, cy -  40, 65, 110,  (100, 140, 180)),   # torso
            (cx - 30, cy + 110, 28, 80,  (90, 130, 170)),  # pierna izq
            (cx + 30, cy + 110, 28, 80,  (90, 130, 170)),  # pierna der
            (cx - 85, cy -  20, 22, 80,  (95, 135, 175)),  # brazo izq
            (cx + 85, cy -  20, 22, 80,  (95, 135, 175)),  # brazo der
        ]
        for bx, by, rx, ry, color in body_parts:
            cv2.ellipse(rgb, (bx, by), (rx, ry), 0, 0, 360, color, -1)

        # Pequeña oscilación para simular video
        offset = int(3 * np.sin(self._frame_n * 0.05))
        rgb = np.roll(rgb, offset, axis=1)

        # ── Depth sintético ───────────────────────────────────────────────────
        depth = np.zeros((h, w), dtype=np.float32)

        # Fondo a 2.5m
        depth[:] = 2.5

        # Cuerpo: elipsoide a 1.5m con variación por parte
        for bx, by, rx, ry, _ in body_parts:
            yy, xx = np.ogrid[:h, :w]
            mask   = ((xx - bx)**2 / max(rx**2, 1) +
                      (yy - by)**2 / max(ry**2, 1)) <= 1.0
            # Profundidad varía como semiesfera (centro más cercano)
            dist_sq = ((xx - bx)**2 / max(rx**2, 1) +
                       (yy - by)**2 / max(ry**2, 1))
            body_depth = 1.5 + 0.15 * dist_sq
            depth[mask] = np.minimum(depth[mask], body_depth[mask])

        # Enmascarar valores fuera del rango válido
        depth[(depth < self.config.depth_min) |
              (depth > self.config.depth_max)] = 0.0

        # Añadir ruido gaussiano leve (simula sensor real)
        noise = np.random.normal(0, 0.005, depth.shape).astype(np.float32)
        depth = np.where(depth > 0, depth + noise, depth)

        return rgb, depth


# ── Script de prueba independiente ───────────────────────────────────────────
if __name__ == "__main__":
    """
    Prueba la cámara en modo simulación mostrando los frames en ventana.

    Para probar con cámara real:
        cam = RealSenseCamera(simulate=False)

    Controles:
        Q / ESC → salir
        S       → guardar frame actual en data/captured/
    """
    import os

    print("=== Test de cámara Body3D ===")
    print("Controles: Q/ESC = salir | S = guardar frame\n")

    cam = RealSenseCamera(simulate=True)
    cam.start()

    os.makedirs("data/captured", exist_ok=True)
    frame_count = 0

    try:
        while True:
            rgb, depth = cam.get_frames()
            depth_vis  = cam.get_depth_colormap(depth)

            # Combinar RGB y Depth side-by-side
            combined = np.hstack([rgb, depth_vis])
            cv2.putText(
                combined,
                f"Frame {frame_count:04d} | "
                f"Depth centro: {depth[depth.shape[0]//2, depth.shape[1]//2]:.2f}m",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

            cv2.imshow("Body3D — RGB | Depth  (Q=salir, S=guardar)", combined)
            frame_count += 1

            key = cv2.waitKey(33) & 0xFF
            if key in (ord('q'), 27):   # Q o ESC
                break
            elif key == ord('s'):
                cv2.imwrite(f"data/captured/rgb_{frame_count:04d}.png", rgb)
                np.save(f"data/captured/depth_{frame_count:04d}.npy", depth)
                print(f"  [S] Frame {frame_count} guardado en data/captured/")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"\nTotal frames procesados: {frame_count}")