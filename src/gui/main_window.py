"""
main_window.py
--------------
GUI Body3D Reconstruction — 3 páginas con navegación por tabs.

Página 1 — Captura:    selector de vistas, carga RGB+depth, sliders ROI/profundidad, datos paciente
Página 2 — Mediciones: líneas de medición ajustables, tabla en tiempo real, reconstrucción 3D
Página 3 — Resultados: comparación SMPL, mapa de volumen, exportar PDF
"""

import sys, os, cv2
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QLineEdit, QComboBox, QGridLayout, QVBoxLayout,
    QHBoxLayout, QSizePolicy, QFrame, QProgressBar, QMessageBox,
    QFileDialog, QScrollArea, QStackedWidget, QSpinBox, QGroupBox,
    QTabWidget, QDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

# ── Paleta ─────────────────────────────────────────────────────────────────────
TEAL        = "#0F6E56"
TEAL_LIGHT  = "#E1F5EE"
CORAL       = "#993C1D"
CORAL_LIGHT = "#FAECE7"
PURPLE      = "#534AB7"
PURPLE_LIGHT= "#EEEDFE"
BLUE        = "#378ADD"
BG_MAIN     = "#F5F5F3"
BG_PANEL    = "#FFFFFF"
BG_CARD     = "#F1EFE8"
BORDER      = "#D3D1C7"
TEXT_PRI    = "#2C2C2A"
TEXT_SEC    = "#5F5E5A"
TEXT_HINT   = "#888780"

ZONES = ["cuello", "pecho", "brazo", "cintura", "cadera", "muslo", "rodilla"]


# ── Helpers de UI ──────────────────────────────────────────────────────────────
def hline():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color:{BORDER};"); return f

def vline():
    f = QFrame(); f.setFrameShape(QFrame.VLine)
    f.setFixedWidth(1); f.setStyleSheet(f"color:{BORDER};"); return f

def section_label(text):
    l = QLabel(text)
    l.setStyleSheet(f"font-size:11px;font-weight:500;color:{TEXT_SEC};")
    return l

def btn(text, fg=TEXT_PRI, bg=BG_CARD):
    b = QPushButton(text)
    b.setStyleSheet(f"""
        QPushButton{{background:{bg};color:{fg};border:0.5px solid {fg};
            border-radius:6px;padding:6px 12px;font-size:12px;}}
        QPushButton:disabled{{background:{BG_CARD};color:{TEXT_HINT};border-color:{BORDER};}}
        QPushButton:hover{{opacity:0.85;}}
    """)
    return b

def input_style():
    return (f"background:{BG_CARD};border:0.5px solid {BORDER};"
            f"border-radius:6px;padding:4px 8px;font-size:12px;color:{TEXT_PRI};")

def labeled(label_text, widget):
    c = QWidget(); l = QVBoxLayout(c)
    l.setContentsMargins(0,0,0,0); l.setSpacing(2)
    lb = QLabel(label_text); lb.setStyleSheet(f"font-size:11px;color:{TEXT_SEC};")
    l.addWidget(lb); l.addWidget(widget); return c


# ── Estado compartido ──────────────────────────────────────────────────────────
class AppState:
    """Estado global compartido entre las 3 páginas."""
    def __init__(self):
        self.n_views    = 1
        self.view_names = ["frontal"]
        self.rgbs       = {}
        self.depths     = {}
        self.roi_params = {}
        self.patient    = {
            "name": "", "sex": "female", "age": 30,
            "weight": 65.0, "height": 1.65, "body_fat": 25.0,
        }
        self.segs         = {}
        self.pcds         = {}
        # ── MODIFICACIÓN 1 ─────────────────────────────────────────────────────
        # STATE.measurements ahora guarda el dict completo de extract_measurements
        # Estructura: {zona: {"y_px": int, "circumference_cm": float,
        #                     "width_mm": float, "delta_mm": float, ...}}
        # Antes era: {zona: float}  ← formato viejo, ya no se usa
        self.measurements = {}
        # ──────────────────────────────────────────────────────────────────────
        self.diag         = {}
        self.height_cm    = 0.0
        self.smpl_result  = None
        self.pipeline_done = False

    def set_n_views(self, n):
        self.n_views = n
        names = ["frontal","posterior","lateral_izq","lateral_der"]
        self.view_names = names[:n]
        defaults = {
            "frontal":     {"x1":396,"x2":755,"y1":0,"y2":626,"d_min":1410,"d_max":1790},
            "posterior":   {"x1":409,"x2":780,"y1":0,"y2":626,"d_min":1180,"d_max":1720},
            "lateral_izq": {"x1":448,"x2":716,"y1":0,"y2":669,"d_min":1090,"d_max":1490},
            "lateral_der": {"x1":422,"x2":729,"y1":0,"y2":626,"d_min":930, "d_max":1750},
        }
        for name in self.view_names:
            if name not in self.roi_params:
                self.roi_params[name] = defaults[name].copy()


STATE = AppState()


# ── Widget imagen con overlay ──────────────────────────────────────────────────
class ImageWithOverlay(QLabel):
    """QLabel que acepta líneas horizontales de medición superpuestas."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background:{BG_CARD};border-radius:6px;")
        self.setMinimumSize(200, 200)
        self._lines   = []
        self._pixmap_base = None

    def set_image_array(self, img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qi = QImage(rgb.data, w, h, c*w, QImage.Format_RGB888)
        self._pixmap_base = QPixmap.fromImage(qi)
        self._redraw()

    def set_lines(self, lines):
        """lines: list of (y_norm 0-1, color_hex, label_str)"""
        self._lines = lines
        self._redraw()

    def _redraw(self):
        if self._pixmap_base is None:
            return
        pix = self._pixmap_base.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        result = QPixmap(pix.size())
        result.fill(Qt.transparent)
        painter = QPainter(result)
        painter.drawPixmap(0, 0, pix)
        h = pix.height()
        w = pix.width()
        for y_norm, color, label in self._lines:
            y_px = int(y_norm * h)
            pen = QPen(QColor(color)); pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(0, y_px, w, y_px)
            painter.setFont(QFont("Arial", 9))
            painter.drawText(4, y_px - 3, label)
        painter.end()
        self.setPixmap(result)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._redraw()


# ── Hilo del pipeline ──────────────────────────────────────────────────────────
class PipelineThread(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def run(self):
        try:
            import sys; sys.path.insert(0,".")
            from src.preprocessing       import preprocess_depth
            from src.segmentation        import segment_body
            from src.reconstruction      import reconstruct_pointcloud
            from src.multi_view_pipeline import combine_measurements
            from src.reconstruction      import FY

            segs = {}; pcds = {}

            for name in STATE.view_names:
                self.progress.emit(f"Procesando {name}...")
                rgb   = STATE.rgbs.get(name)
                depth = STATE.depths.get(name)
                if rgb is None or depth is None:
                    continue
                roi = STATE.roi_params[name]
                depth_clean = preprocess_depth(depth)
                seg = segment_body(
                    rgb, depth_clean,
                    d_min_mm=roi["d_min"], d_max_mm=roi["d_max"],
                    x_min=roi["x1"], x_max=roi["x2"],
                    y_min=roi["y1"], y_max=roi["y2"],
                )
                pcd = reconstruct_pointcloud(seg["depth_body"], seg["rgb_body"])
                segs[name] = seg
                pcds[name]  = pcd

            STATE.segs = segs
            STATE.pcds  = pcds

            self.progress.emit("Extrayendo medidas...")
            if "frontal" in segs:
                from src.multi_view_pipeline import (
                    combine_measurements, ANATOMICAL_NORM
                )
                dummy = {"mask": segs.get("frontal",{}).get("mask"),
                         "depth_body": segs.get("frontal",{}).get("depth_body"),
                         "y_top": 0, "y_bottom": 1}

                def _mk(name):
                    s = segs.get(name, {})
                    m = s.get("mask")
                    d = s.get("depth_body")
                    if m is None:
                        return dummy
                    rows = np.where(m.sum(axis=1) > 10)[0]
                    yt = int(rows.min()) if len(rows) > 0 else 0
                    yb = int(rows.max()) if len(rows) > 0 else m.shape[0]
                    return {"mask": m, "depth_body": d,
                            "y_top": yt, "y_bottom": yb}

                # ── MODIFICACIÓN 2 ─────────────────────────────────────────────
                # Se pasa rgb_image="frontal" para que combine_measurements
                # (y dentro de él extract_measurements) active MediaPipe
                # y calcule y_px automáticamente para cada zona.
                rgb_frontal = STATE.rgbs.get("frontal")

                meas = combine_measurements(
                    _mk("frontal"), _mk("posterior"),
                    _mk("lateral_izq"), _mk("lateral_der"),
                    rgb_image=rgb_frontal,   # ← activa MediaPipe
                )

                # STATE.measurements guarda el dict completo (no solo circ_cm)
                # Estructura: {zona: {"y_px", "circumference_cm", "width_mm", ...}}
                STATE.measurements = meas

                # diag sigue igual para Page3
                STATE.diag = {k: {
                    "w_front_cm": round(v["width_mm"]/10,1) if v else None,
                    "w_side_cm":  round(v["depth_mm"]/10,1) if v else None,
                    "perim_cm":   v["circumference_cm"] if v else None,
                } for k, v in meas.items() if v}
                # ──────────────────────────────────────────────────────────────

                # Altura
                seg_f = segs["frontal"]
                rows  = np.where(seg_f["mask"].sum(axis=1) > 10)[0]
                hp    = int(rows.max()-rows.min()) if len(rows) > 0 else 0
                d_ref = float(np.median(seg_f["depth_body"]
                              [seg_f["depth_body"] > 0]))
                STATE.height_cm = hp * d_ref / FY / 10.0

            STATE.pipeline_done = True
            self.finished.emit({"ok": True})

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — CAPTURA
# ══════════════════════════════════════════════════════════════════════════════
class Page1Capture(QWidget):

    go_next = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._view_widgets = {}
        self._roi_sliders  = {}
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12,12,12,12)
        root.setSpacing(10)

        left = QWidget(); left.setFixedWidth(260)
        left.setStyleSheet(f"background:{BG_PANEL};border-radius:8px;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12,12,12,12); ll.setSpacing(8)

        ll.addWidget(section_label("Número de vistas"))
        self.cmb_views = QComboBox()
        self.cmb_views.addItems(["1 vista (frontal)",
                                  "2 vistas (frontal + posterior)",
                                  "3 vistas (+ lateral izq.)",
                                  "4 vistas (completo)"])
        self.cmb_views.setStyleSheet(input_style())
        self.cmb_views.currentIndexChanged.connect(self._on_n_views_changed)
        ll.addWidget(self.cmb_views)

        ll.addWidget(hline())
        ll.addWidget(section_label("Datos del paciente"))

        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText("Nombre del paciente")
        self.inp_name.setStyleSheet(input_style())
        ll.addWidget(labeled("Nombre", self.inp_name))

        r1 = QHBoxLayout()
        self.cmb_sex = QComboBox()
        self.cmb_sex.addItems(["Femenino","Masculino"])
        self.cmb_sex.setStyleSheet(input_style())
        self.inp_age = QLineEdit("30"); self.inp_age.setStyleSheet(input_style())
        r1.addWidget(labeled("Sexo", self.cmb_sex))
        r1.addWidget(labeled("Edad", self.inp_age))
        ll.addLayout(r1)

        r2 = QHBoxLayout()
        self.inp_weight = QLineEdit("65.0"); self.inp_weight.setStyleSheet(input_style())
        self.inp_height = QLineEdit("1.65"); self.inp_height.setStyleSheet(input_style())
        r2.addWidget(labeled("Peso (kg)", self.inp_weight))
        r2.addWidget(labeled("Talla (m)", self.inp_height))
        ll.addLayout(r2)

        bf_row = QHBoxLayout()
        self.slider_bf = QSlider(Qt.Horizontal); self.slider_bf.setRange(5,45); self.slider_bf.setValue(25)
        self.lbl_bf = QLabel("25%"); self.lbl_bf.setFixedWidth(32)
        self.lbl_bf.setStyleSheet(f"font-size:12px;color:{TEXT_PRI};")
        self.slider_bf.valueChanged.connect(lambda v: self.lbl_bf.setText(f"{v}%"))
        bf_row.addWidget(self.slider_bf); bf_row.addWidget(self.lbl_bf)
        ll.addWidget(labeled("% grasa corporal", QWidget()))
        ll.addLayout(bf_row)

        ll.addWidget(hline())

        self.btn_cam_start = btn("Iniciar cámara", TEAL, TEAL_LIGHT)
        self.btn_cam_stop  = btn("Detener cámara", CORAL, CORAL_LIGHT)
        self.btn_cam_stop.setEnabled(False)
        self.btn_cam_start.clicked.connect(self._start_camera)
        self.btn_cam_stop.clicked.connect(self._stop_camera)
        ll.addWidget(self.btn_cam_start)
        ll.addWidget(self.btn_cam_stop)

        self.btn_capture = btn("Capturar frame", TEAL, TEAL_LIGHT)
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self._capture_frame)
        ll.addWidget(self.btn_capture)

        self.btn_import = btn("Importar imágenes")
        self.btn_import.clicked.connect(self._import_images)
        ll.addWidget(self.btn_import)

        ll.addWidget(hline())

        self.btn_landmarks_p1 = btn("Detectar landmarks (MediaPipe)", BLUE, "#E6F1FB")
        self.btn_landmarks_p1.clicked.connect(self._detect_landmarks_p1)
        ll.addWidget(self.btn_landmarks_p1)

        self.lbl_status = QLabel("En espera")
        self.lbl_status.setStyleSheet(f"font-size:11px;color:{TEXT_HINT};")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        ll.addStretch()

        self.btn_next = btn("Continuar a Mediciones →", PURPLE, PURPLE_LIGHT)
        self.btn_next.clicked.connect(self._save_patient_and_next)
        ll.addWidget(self.btn_next)

        root.addWidget(left)

        self.center_scroll = QScrollArea()
        self.center_scroll.setWidgetResizable(True)
        self.center_scroll.setStyleSheet("border:none;background:transparent;")
        self.center_widget = QWidget()
        self.center_layout = QVBoxLayout(self.center_widget)
        self.center_layout.setSpacing(12)
        self.center_scroll.setWidget(self.center_widget)
        root.addWidget(self.center_scroll, stretch=1)

        self._on_n_views_changed(0)

        self.cam = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def _on_n_views_changed(self, idx):
        STATE.set_n_views(idx + 1)
        self._rebuild_view_widgets()

    def _rebuild_view_widgets(self):
        while self.center_layout.count():
            item = self.center_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._view_widgets = {}
        self._roi_sliders  = {}

        grid_container = QWidget()
        grid_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer_grid = QGridLayout(grid_container)
        outer_grid.setSpacing(8)
        outer_grid.setContentsMargins(0,0,0,0)
        outer_grid.setColumnStretch(0, 1)
        outer_grid.setColumnStretch(1, 1)
        outer_grid.setRowStretch(0, 1)
        outer_grid.setRowStretch(1, 1)
        positions = [(0,0),(0,1),(1,0),(1,1)]

        for idx, name in enumerate(STATE.view_names):
            row_g, col_g = positions[idx]

            group = QGroupBox(name.replace("_"," ").capitalize())
            group.setStyleSheet(f"""
                QGroupBox{{background:{BG_PANEL};border:0.5px solid {BORDER};
                    border-radius:8px;margin-top:8px;font-size:11px;font-weight:500;}}
                QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 3px;}}
            """)
            gl = QVBoxLayout(group)
            gl.setContentsMargins(6,10,6,6)
            gl.setSpacing(4)

            img_row = QHBoxLayout()
            img_row.setSpacing(4)
            w_rgb = ImageWithOverlay()
            w_dep = ImageWithOverlay()
            w_rgb.setMinimumHeight(100)
            w_dep.setMinimumHeight(100)
            w_rgb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w_dep.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w_rgb.setText("RGB")
            w_dep.setText("Depth")

            rgb_col = QVBoxLayout()
            rgb_lbl = QLabel("RGB")
            rgb_lbl.setStyleSheet(f"font-size:9px;color:{TEXT_HINT};")
            rgb_col.setSpacing(1); rgb_col.addWidget(rgb_lbl); rgb_col.addWidget(w_rgb)

            dep_col = QVBoxLayout()
            dep_lbl = QLabel("Depth")
            dep_lbl.setStyleSheet(f"font-size:9px;color:{TEXT_HINT};")
            dep_col.setSpacing(1); dep_col.addWidget(dep_lbl); dep_col.addWidget(w_dep)

            img_row.addLayout(rgb_col)
            img_row.addLayout(dep_col)
            gl.addLayout(img_row)

            roi = STATE.roi_params[name]
            sliders = {}
            h_img, w_img = 720, 1280

            params_def = [
                ("x1%",  "x1",   int(roi["x1"]/w_img*100), 0,   60),
                ("x2%",  "x2",   int(roi["x2"]/w_img*100), 40, 100),
                ("y1%",  "y1",   int(roi["y1"]/h_img*100), 0,   50),
                ("y2%",  "y2",   int(roi["y2"]/h_img*100), 50, 100),
                ("dMin", "d_min",roi["d_min"]//10,          50, 250),
                ("dMax", "d_max",roi["d_max"]//10,          50, 400),
            ]

            sgrid = QGridLayout()
            sgrid.setSpacing(2)
            sgrid.setContentsMargins(0,0,0,0)

            for row_i, (label_txt, key, val, lo, hi) in enumerate(params_def):
                lbl = QLabel(label_txt)
                lbl.setStyleSheet(f"font-size:9px;color:{TEXT_SEC};min-width:32px;")
                sl  = QSlider(Qt.Horizontal)
                sl.setRange(lo, hi); sl.setValue(val)
                sl.setFixedHeight(16)
                val_lbl = QLabel(str(val))
                val_lbl.setFixedWidth(28)
                val_lbl.setStyleSheet(f"font-size:9px;color:{TEXT_PRI};")

                def _make_cb(k, vl, nm):
                    def cb(v):
                        vl.setText(str(v))
                        if k in ("x1","x2"):
                            STATE.roi_params[nm][k] = int(v/100*1280)
                        elif k in ("y1","y2"):
                            STATE.roi_params[nm][k] = int(v/100*720)
                        else:
                            STATE.roi_params[nm][k] = v * 10
                        self._update_overlay(nm)
                    return cb

                sl.valueChanged.connect(_make_cb(key, val_lbl, name))
                sliders[key] = sl
                sgrid.addWidget(lbl,     row_i, 0)
                sgrid.addWidget(sl,      row_i, 1)
                sgrid.addWidget(val_lbl, row_i, 2)

            gl.addLayout(sgrid)

            self._view_widgets[name] = {"rgb": w_rgb, "depth": w_dep}
            self._roi_sliders[name]  = sliders
            outer_grid.addWidget(group, row_g, col_g)

        for i in range(len(STATE.view_names), 4):
            row_g, col_g = positions[i]
            outer_grid.addWidget(QWidget(), row_g, col_g)

        self.center_layout.addWidget(grid_container, stretch=1)

    def _update_overlay(self, name):
        depth = STATE.depths.get(name)
        if depth is None:
            return
        roi = STATE.roi_params[name]
        valid = depth[(depth > 300) & (depth < 4000)]
        if len(valid) == 0:
            return
        d_norm = np.zeros_like(depth, dtype=np.uint8)
        mask_v = (depth > 300) & (depth < 4000)
        d_norm[mask_v] = ((depth[mask_v] - valid.min()) /
                           (valid.max() - valid.min()) * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
        cv2.rectangle(depth_vis,
                      (roi["x1"], roi["y1"]), (roi["x2"], roi["y2"]),
                      (0, 255, 0), 2)
        self._view_widgets[name]["depth"].set_image_array(depth_vis)

    def _start_camera(self):
        from src.camera import RealSenseCamera, CameraConfig
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            if len(ctx.devices) == 0: raise RuntimeError
            self.cam = RealSenseCamera(simulate=False, config=CameraConfig())
            mode = "RealSense D455"
        except Exception:
            self.cam = RealSenseCamera(simulate=True, config=CameraConfig())
            mode = "Simulación"
        self.cam.start()
        self.timer.start(33)
        self.btn_cam_start.setEnabled(False)
        self.btn_cam_stop.setEnabled(True)
        self.btn_capture.setEnabled(True)
        self.lbl_status.setText(f"Cámara activa — {mode}")
        self._current_cam_view = STATE.view_names[0]

    def _stop_camera(self):
        self.timer.stop()
        if self.cam: self.cam.stop(); self.cam = None
        self.btn_cam_start.setEnabled(True)
        self.btn_cam_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.lbl_status.setText("Cámara detenida")

    def _update_frame(self):
        if not self.cam: return
        try:
            rgb, depth = self.cam.get_frames()
            name = getattr(self, "_current_cam_view", STATE.view_names[0])
            if name in self._view_widgets:
                self._view_widgets[name]["rgb"].set_image_array(rgb)
            self._last_cam_rgb   = rgb
            self._last_cam_depth = depth
        except Exception as e:
            self.lbl_status.setText(f"Error: {e}")
            self._stop_camera()

    def _capture_frame(self):
        rgb   = getattr(self, "_last_cam_rgb",   None)
        depth = getattr(self, "_last_cam_depth", None)
        if rgb is None: return
        name = getattr(self, "_current_cam_view", STATE.view_names[0])
        STATE.rgbs[name]   = rgb
        STATE.depths[name] = depth
        self._update_overlay(name)
        self.lbl_status.setText(f"Frame {name} capturado ✓")

    def _import_images(self):
        data_dir = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta con imágenes",
            str(Path.home())
        )
        if not data_dir:
            return

        data_dir = Path(data_dir)
        loaded = []
        for name in STATE.view_names:
            rgb_path   = data_dir / f"{name}_rgb.png"
            depth_path = data_dir / f"{name}_depth.npy"
            if rgb_path.exists() and depth_path.exists():
                STATE.rgbs[name]   = cv2.imread(str(rgb_path))
                STATE.depths[name] = np.load(str(depth_path)).astype("float32")
                self._view_widgets[name]["rgb"].set_image_array(STATE.rgbs[name])
                self._update_overlay(name)
                loaded.append(name)

        if loaded:
            self.lbl_status.setText(f"Cargadas: {', '.join(loaded)} ✓")
        else:
            self.lbl_status.setText("No se encontraron archivos con los nombres esperados.")
            self._import_one_by_one()

    def _import_one_by_one(self):
        for name in STATE.view_names:
            rgb_path, _ = QFileDialog.getOpenFileName(
                self, f"RGB — {name}", str(Path.home()),
                "Imágenes (*.png *.jpg);;Todas (*.*)"
            )
            if not rgb_path: continue
            depth_path, _ = QFileDialog.getOpenFileName(
                self, f"Depth .npy — {name}", str(Path(rgb_path).parent),
                "NumPy (*.npy);;Todas (*.*)"
            )
            STATE.rgbs[name]   = cv2.imread(rgb_path)
            if depth_path:
                STATE.depths[name] = np.load(depth_path).astype("float32")
            self._view_widgets[name]["rgb"].set_image_array(STATE.rgbs[name])
            self._update_overlay(name)
        self.lbl_status.setText("Importación completada ✓")

    def _detect_landmarks_p1(self):
        """
        Botón 'Detectar landmarks' en Página 1.
        Corre MediaPipe sobre todas las vistas cargadas y actualiza
        las imágenes RGB y depth con los 33 landmarks dibujados.
        """
        from src.pose_overlay import draw_landmarks_on_images

        if not STATE.rgbs:
            self.lbl_status.setText("Carga imágenes primero.")
            return

        self.btn_landmarks_p1.setEnabled(False)
        self.lbl_status.setText("Detectando landmarks...")
        QApplication.processEvents()

        found_any = False
        for name in STATE.view_names:
            rgb   = STATE.rgbs.get(name)
            depth = STATE.depths.get(name)
            if rgb is None:
                continue
            if depth is None:
                depth = np.zeros(rgb.shape[:2], dtype=np.float32)

            rgb_ann, depth_ann, ok = draw_landmarks_on_images(rgb, depth)

            if ok and name in self._view_widgets:
                self._view_widgets[name]["rgb"].set_image_array(rgb_ann)
                self._view_widgets[name]["depth"].set_image_array(depth_ann)
                found_any = True

        self.btn_landmarks_p1.setEnabled(True)
        self.lbl_status.setText("Landmarks detectados ✓" if found_any
                                 else "No se detectó ninguna persona.")

    def _save_patient_and_next(self):
        STATE.patient["name"]     = self.inp_name.text().strip() or "No especificado"
        STATE.patient["sex"]      = "female" if self.cmb_sex.currentIndex() == 0 else "male"
        STATE.patient["age"]      = float(self.inp_age.text() or 30)
        STATE.patient["weight"]   = float(self.inp_weight.text() or 65)
        STATE.patient["height"]   = float(self.inp_height.text() or 1.65)
        STATE.patient["body_fat"] = float(self.slider_bf.value())

        if not STATE.rgbs:
            QMessageBox.warning(self, "Sin imágenes",
                                "Carga al menos una imagen antes de continuar.")
            return
        self.go_next.emit()


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — MEDICIONES
# ══════════════════════════════════════════════════════════════════════════════

class ZoneMarkerDialog(QDialog):
    """
    Ventana auxiliar grande para marcar zonas anatómicas.
    Panel izquierdo: imagen grande con rectángulos interactivos.
    Panel derecho: sliders ROI + lista de zonas pendientes.
    """

    ZONE_COLORS = {
        "cuello":      (220, 60,  60),
        "pecho":       (255, 144, 30),
        "brazo_izq":   (180, 180, 0),
        "brazo_der":   (120, 180, 0),
        "cintura":     (0,   200, 0),
        "cadera":      (0,   150, 220),
        "muslo":       (200, 0,   200),
        "rodilla":     (60,  60,  220),
        "prof_pecho":  (255, 165, 0),
        "prof_cintura":(200, 100, 0),
    }

    def __init__(self, view_name, image_rgb, zones_for_view,
                 existing_zones=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Marcar zonas — {view_name}")
        self.setModal(True)
        self.resize(1200, 700)

        self.view_name      = view_name
        self.zones_for_view = zones_for_view
        self.zone_rects     = dict(existing_zones or {})
        self.current_idx    = sum(1 for z in zones_for_view if z in self.zone_rects)
        self.first_pt       = None
        self.mouse_pos      = None
        self._orig_rgb      = image_rgb.copy()

        # Usar depth como imagen base del canvas (en vez del RGB)
        depth_src = STATE.depths.get(view_name)
        if depth_src is not None:
            valid = depth_src[(depth_src > 300) & (depth_src < 4000)]
            d_norm = np.zeros_like(depth_src, dtype=np.uint8)
            if len(valid) > 0:
                mask_v = (depth_src > 300) & (depth_src < 4000)
                d_norm[mask_v] = ((depth_src[mask_v]-valid.min())/(valid.max()-valid.min())*255).astype(np.uint8)
            base_img = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
            # Guardar como RGB para consistencia con _orig_rgb
            self._orig_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        else:
            self._orig_rgb = image_rgb.copy()

        h, w = self._orig_rgb.shape[:2]
        # Limitar tamaño del canvas para que quepan los botones del panel derecho
        # Máximo 700px ancho y 550px alto
        # El canvas se adapta al espacio disponible — panel derecho ocupa 360px
        # La ventana es 1800px ancho, panel derecho 360px, margenes ~20px
        available_w = 1800 - 360 - 40
        available_h = 900 - 80   # restar barra titulo e instruccion
        self._scale = min(available_w / w, available_h / h)
        nw = int(w * self._scale)
        nh = int(h * self._scale)
        scaled = cv2.resize(self._orig_rgb, (nw, nh))
        qi = QImage(scaled.data, nw, nh, 3*nw, QImage.Format_RGB888).copy()
        self._base_pixmap = QPixmap.fromImage(qi)
        self._img_w = nw
        self._img_h = nh

        self._build()
        self.setMouseTracking(True)

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left = QWidget()
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(0,0,0,0); ll.setSpacing(4)

        self.lbl_instr = QLabel("")
        self.lbl_instr.setFixedHeight(36)
        self.lbl_instr.setStyleSheet(
            f"font-size:13px;font-weight:500;padding:6px;"
            f"border-radius:6px;background:{BG_CARD};color:{TEXT_PRI};"
        )
        ll.addWidget(self.lbl_instr)

        # Solo canvas depth — interactivo para marcar zonas
        dep_col = QVBoxLayout(); dep_col.setSpacing(2)
        dep_title = QLabel("Depth")
        dep_title.setStyleSheet(f"font-size:10px;color:{TEXT_HINT};")
        ll.addWidget(dep_title)

        # canvas es el depth — interactivo para marcar zonas con clic
        self.canvas = QLabel()
        self.canvas.setFixedSize(self._img_w, self._img_h)
        self.canvas.setStyleSheet(f"background:{BG_CARD};border-radius:6px;")
        self.canvas.setMouseTracking(True)
        self.canvas.mousePressEvent = self._on_click
        self.canvas.mouseMoveEvent  = self._on_move
        ll.addWidget(self.canvas)
        ll.addStretch()
        root.addWidget(left, stretch=0)

        right = QWidget()
        right.setFixedWidth(340)
        right.setStyleSheet(f"background:{BG_PANEL};border-radius:8px;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12,12,12,12); rl.setSpacing(8)

        rl.addWidget(section_label(f"Vista: {self.view_name}"))
        rl.addWidget(hline())

        rl.addWidget(section_label("Ajuste ROI y profundidad"))
        roi = STATE.roi_params[self.view_name]
        self._roi_sliders = {}
        h_img, w_img = 720, 1280

        params = [
            ("x1 %",  "x1",   int(roi["x1"]/w_img*100), 0,   60),
            ("x2 %",  "x2",   int(roi["x2"]/w_img*100), 40, 100),
            ("y1 %",  "y1",   int(roi["y1"]/h_img*100), 0,   50),
            ("y2 %",  "y2",   int(roi["y2"]/h_img*100), 50, 100),
            ("d_min", "d_min",roi["d_min"]//10,          50, 250),
            ("d_max", "d_max",roi["d_max"]//10,          50, 400),
        ]
        for label_txt, key, val, lo, hi in params:
            row = QHBoxLayout()
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(f"font-size:10px;color:{TEXT_SEC};min-width:48px;")
            sl  = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi); sl.setValue(val)
            vl  = QLabel(str(val))
            vl.setFixedWidth(28)
            vl.setStyleSheet(f"font-size:10px;color:{TEXT_PRI};")
            def _cb(v, k=key, vlbl=vl):
                vlbl.setText(str(v))
                if k in ("x1","x2"):
                    STATE.roi_params[self.view_name][k] = int(v/100*1280)
                elif k in ("y1","y2"):
                    STATE.roi_params[self.view_name][k] = int(v/100*720)
                else:
                    STATE.roi_params[self.view_name][k] = v * 10
                self._refresh_base()
            sl.valueChanged.connect(_cb)
            self._roi_sliders[key] = sl
            row.addWidget(lbl); row.addWidget(sl); row.addWidget(vl)
            rl.addLayout(row)

        btn_apply = btn("Aplicar ROI", TEAL, TEAL_LIGHT)
        btn_apply.clicked.connect(self._refresh_base)
        rl.addWidget(btn_apply)
        rl.addWidget(hline())

        rl.addWidget(section_label("Zonas a marcar"))
        self._zone_list_labels = {}
        for z in self.zones_for_view:
            c = self.ZONE_COLORS.get(z, (180,180,180))
            row = QHBoxLayout()
            dot = QLabel()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(f"background:rgb({c[0]},{c[1]},{c[2]});border-radius:6px;")
            lbl = QLabel(z)
            lbl.setStyleSheet(f"font-size:11px;color:{TEXT_SEC};")
            status = QLabel("✓" if z in self.zone_rects else "○")
            status.setStyleSheet(
                f"font-size:11px;color:{'#0F6E56' if z in self.zone_rects else TEXT_HINT};"
            )
            row.addWidget(dot); row.addWidget(lbl); row.addStretch()
            row.addWidget(status)
            rl.addLayout(row)
            self._zone_list_labels[z] = status

        rl.addWidget(hline())
        rl.addStretch()

        btn_landmarks = btn("Detectar landmarks (MediaPipe)", BLUE, "#E6F1FB")
        btn_landmarks.clicked.connect(self._detect_landmarks)
        rl.addWidget(btn_landmarks)

        btn_redo = btn("↩ Rehacer última", TEXT_SEC, BG_CARD)
        btn_redo.clicked.connect(self._redo)
        rl.addWidget(btn_redo)

        btn_save = btn("Guardar zonas", TEAL, TEAL_LIGHT)
        btn_save.clicked.connect(self.accept)
        rl.addWidget(btn_save)

        btn_cancel = btn("Cancelar", CORAL, CORAL_LIGHT)
        btn_cancel.clicked.connect(self.reject)
        rl.addWidget(btn_cancel)

        root.addWidget(right, stretch=0)

        self._refresh()

    def _refresh_base(self):
        """Regenera el canvas depth con los parametros ROI actuales."""
        depth = STATE.depths.get(self.view_name)
        if depth is None:
            return
        valid = depth[(depth > 300) & (depth < 4000)]
        d_norm = np.zeros_like(depth, dtype=np.uint8)
        if len(valid) > 0:
            mask_v = (depth > 300) & (depth < 4000)
            d_norm[mask_v] = ((depth[mask_v]-valid.min())/(valid.max()-valid.min())*255).astype(np.uint8)
        base_bgr = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
        # Dibujar ROI box sobre el depth
        roi = STATE.roi_params[self.view_name]
        cv2.rectangle(base_bgr, (roi["x1"],roi["y1"]),
                      (roi["x2"],roi["y2"]), (0,255,255), 2)
        self._orig_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
        h, w = self._orig_rgb.shape[:2]
        scale = min(700/w, 550/h)
        nw, nh = int(w*scale), int(h*scale)
        self._scale = scale
        scaled = cv2.resize(self._orig_rgb, (nw, nh))
        qi = QImage(scaled.data, nw, nh, 3*nw, QImage.Format_RGB888).copy()
        self._base_pixmap = QPixmap.fromImage(qi)
        self.canvas.setFixedSize(nw, nh)
        self._img_w = nw
        self._img_h = nh
        self._refresh()

    def _refresh(self):
        pix     = QPixmap(self._base_pixmap)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)

        for z, r in self.zone_rects.items():
            c = self.ZONE_COLORS.get(z, (180,180,180))
            pen = QPen(QColor(*c), 2)
            painter.setPen(pen)
            x1s = int(r["x1"]*self._scale)
            y1s = int(r["y1"]*self._scale)
            x2s = int(r["x2"]*self._scale)
            y2s = int(r["y2"]*self._scale)
            painter.drawRect(x1s, y1s, x2s-x1s, y2s-y1s)
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            painter.setPen(QPen(QColor(*c)))
            painter.drawText(x1s+3, y1s+14, z)

        if self.first_pt and self.mouse_pos and self.current_idx < len(self.zones_for_view):
            z = self.zones_for_view[self.current_idx]
            c = self.ZONE_COLORS.get(z, (255,255,255))
            x1 = min(self.first_pt.x(), self.mouse_pos.x())
            y1 = min(self.first_pt.y(), self.mouse_pos.y())
            x2 = max(self.first_pt.x(), self.mouse_pos.x())
            y2 = max(self.first_pt.y(), self.mouse_pos.y())
            pen = QPen(QColor(*c), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            h_orig = int((y2-y1)/self._scale)
            w_orig = int((x2-x1)/self._scale)
            painter.setFont(QFont("Arial", 8))
            painter.setPen(QPen(QColor(*c)))
            painter.drawText(x1, y2+14, f"w={w_orig}px  h={h_orig}px")

        painter.end()
        self.canvas.setPixmap(pix)

        if self.current_idx < len(self.zones_for_view):
            z = self.zones_for_view[self.current_idx]
            c = self.ZONE_COLORS.get(z, (180,180,180))
            n = self.current_idx + 1
            total = len(self.zones_for_view)
            msg = (f"Zona {n}/{total} — {z}: "
                   f"{'clic esquina inferior-derecha' if self.first_pt else 'clic esquina superior-izquierda'}")
            self.lbl_instr.setText(msg)
            self.lbl_instr.setStyleSheet(
                f"font-size:13px;font-weight:500;padding:6px;border-radius:6px;"
                f"background:rgb({max(0,c[0]-180)},{max(0,c[1]-180)},{max(0,c[2]-180)});"
                f"color:rgb({c[0]},{c[1]},{c[2]});"
            )
        else:
            self.lbl_instr.setText(
                f"Todas marcadas ({len(self.zone_rects)}) — presiona Guardar"
            )
            self.lbl_instr.setStyleSheet(
                f"font-size:13px;font-weight:500;padding:6px;"
                f"border-radius:6px;background:{TEAL_LIGHT};color:{TEAL};"
            )

        for z, lbl in self._zone_list_labels.items():
            if z in self.zone_rects:
                lbl.setText("✓")
                lbl.setStyleSheet(f"font-size:11px;color:#0F6E56;font-weight:500;")
            else:
                lbl.setText("○")
                lbl.setStyleSheet(f"font-size:11px;color:{TEXT_HINT};")

    def _on_click(self, e):
        if e.button() != Qt.LeftButton:
            return
        if self.current_idx >= len(self.zones_for_view):
            return
        pos = e.pos()
        if self.first_pt is None:
            self.first_pt = pos
        else:
            x1s = min(self.first_pt.x(), pos.x())
            y1s = min(self.first_pt.y(), pos.y())
            x2s = max(self.first_pt.x(), pos.x())
            y2s = max(self.first_pt.y(), pos.y())
            if y2s - y1s < 15:
                mid = (y1s+y2s)//2
                y1s, y2s = mid-15, mid+15
            x1 = int(x1s/self._scale)
            y1 = int(y1s/self._scale)
            x2 = int(x2s/self._scale)
            y2 = int(y2s/self._scale)
            z = self.zones_for_view[self.current_idx]
            self.zone_rects[z] = {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
            self.first_pt = None
            self.current_idx += 1
        self._refresh()

    def _on_move(self, e):
        self.mouse_pos = e.pos()
        self._refresh()

    def _detect_landmarks(self):
        """
        Detecta los 33 landmarks de MediaPipe y los dibuja sobre
        el canvas RGB y el canvas depth de la subventana de zonas.
        Los landmarks ayudan a ubicar visualmente las zonas anatomicas.
        """
        from src.pose_overlay import (
            _detect_landmarks, _landmarks_to_pixels,
            _draw_on_frame, _depth_to_bgr,
            PARALLAX_X, PARALLAX_Y
        )
        import cv2 as _cv2
        from PyQt5.QtGui import QImage, QPixmap

        rgb   = STATE.rgbs.get(self.view_name)
        depth = STATE.depths.get(self.view_name)
        if rgb is None:
            return

        # Detectar landmarks sobre el RGB (en formato RGB para MediaPipe)
        rgb_for_mp = _cv2.cvtColor(rgb.astype("uint8"), _cv2.COLOR_BGR2RGB)
        landmarks  = _detect_landmarks(rgb_for_mp)
        if landmarks is None:
            return

        H_orig, W_orig = rgb.shape[:2]

        # Puntos para RGB (sin offset)
        points = _landmarks_to_pixels(landmarks, H_orig, W_orig)

        # Puntos para depth (con offset de paralaje)
        points_depth = []
        for (x, y, vis) in points:
            x_d = max(0, min(W_orig - 1, x + PARALLAX_X))
            y_d = max(0, min(H_orig - 1, y + PARALLAX_Y))
            points_depth.append((x_d, y_d, vis))

        H_canvas, W_canvas = self._img_h, self._img_w

        # ── Actualizar canvas depth (unico canvas) con landmarks y paralaje ────
        if depth is not None:
            depth_bgr     = _depth_to_bgr(depth)
            depth_ann_bgr = _draw_on_frame(depth_bgr, points_depth)
            depth_ann_rgb = _cv2.cvtColor(depth_ann_bgr, _cv2.COLOR_BGR2RGB)
            scaled        = _cv2.resize(depth_ann_rgb, (W_canvas, H_canvas))
            qi = QImage(scaled.data, W_canvas, H_canvas,
                        3*W_canvas, QImage.Format_RGB888).copy()
            self._base_pixmap = QPixmap.fromImage(qi)
            # Actualizar _orig_rgb para que _refresh conserve los landmarks
            self._orig_rgb = _cv2.cvtColor(
                _cv2.resize(depth_ann_bgr, (W_orig, H_orig)),
                _cv2.COLOR_BGR2RGB
            )
        self._refresh()

    def _redo(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            z = self.zones_for_view[self.current_idx]
            self.zone_rects.pop(z, None)
            self.first_pt = None
            self._refresh()


class Page2Measurements(QWidget):

    go_next = pyqtSignal()
    go_prev = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._zone_labels  = {}
        self._view_images  = {}
        self._thread       = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12,12,12,12)
        root.setSpacing(10)

        left = QWidget(); left.setFixedWidth(260)
        left.setStyleSheet(f"background:{BG_PANEL};border-radius:8px;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12,12,12,12); ll.setSpacing(6)

        ll.addWidget(hline())

        ll.addWidget(section_label("Medidas calculadas"))
        self.meas_grid = QGridLayout(); self.meas_grid.setSpacing(3)
        self._meas_labels = {}
        for i, zone in enumerate(ZONES):
            z_lbl = QLabel(zone.capitalize())
            z_lbl.setStyleSheet(f"font-size:11px;color:{TEXT_SEC};")
            v_lbl = QLabel("— cm")
            v_lbl.setStyleSheet(f"font-size:12px;font-weight:500;color:{TEXT_PRI};")
            self.meas_grid.addWidget(z_lbl, i, 0)
            self.meas_grid.addWidget(v_lbl, i, 1)
            self._meas_labels[zone] = v_lbl
        ll.addLayout(self.meas_grid)

        self.lbl_height = QLabel("Altura: — cm")
        self.lbl_height.setStyleSheet(f"font-size:11px;color:{TEXT_SEC};margin-top:4px;")
        ll.addWidget(self.lbl_height)

        ll.addWidget(hline())

        self.btn_run = btn("Ejecutar pipeline", TEAL, TEAL_LIGHT)
        self.btn_run.clicked.connect(self._run_pipeline)
        ll.addWidget(self.btn_run)

        self.btn_landmarks_p2 = btn("Detectar landmarks (MediaPipe)", BLUE, "#E6F1FB")
        self.btn_landmarks_p2.clicked.connect(self._detect_landmarks_p2)
        ll.addWidget(self.btn_landmarks_p2)

        self.lbl_status = QLabel("Esperando datos...")
        self.lbl_status.setStyleSheet(f"font-size:11px;color:{TEXT_HINT};")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        ll.addStretch()

        nav = QHBoxLayout()
        self.btn_prev = btn("← Captura", TEXT_SEC, BG_CARD)
        self.btn_next = btn("Resultados →", PURPLE, PURPLE_LIGHT)
        self.btn_prev.clicked.connect(self.go_prev.emit)
        self.btn_next.clicked.connect(self.go_next.emit)
        self.btn_next.setEnabled(False)
        nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next)
        ll.addLayout(nav)

        root.addWidget(left)

        self.img_scroll = QScrollArea()
        self.img_scroll.setWidgetResizable(True)
        self.img_scroll.setStyleSheet("border:none;background:transparent;")
        self.img_container = QWidget()
        self.img_layout    = QVBoxLayout(self.img_container)
        self.img_layout.setSpacing(8)
        self.img_layout.setContentsMargins(0,0,0,0)
        self.img_scroll.setWidget(self.img_container)
        root.addWidget(self.img_scroll, stretch=1)

    def showEvent(self, e):
        super().showEvent(e)
        self._rebuild_images()
        self._update_measurements_display()

    def _rebuild_images(self):
        while self.img_layout.count():
            item = self.img_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._view_images = {}

        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setSpacing(8)

        positions = [(0,0), (0,1), (1,0), (1,1)]

        for i, name in enumerate(STATE.view_names):
            row, col = positions[i]
            cell = QWidget()
            cl   = QVBoxLayout(cell); cl.setContentsMargins(0,0,0,0); cl.setSpacing(4)

            lbl = section_label(name.replace("_"," ").capitalize())
            cl.addWidget(lbl)

            # Solo panel depth
            w_dep = ImageWithOverlay()
            w_dep.setMinimumHeight(320)
            w_dep.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            dep_lbl = QLabel("Depth")
            dep_lbl.setStyleSheet(f"font-size:9px;color:{TEXT_HINT};")
            cl.addWidget(dep_lbl)
            cl.addWidget(w_dep, stretch=1)

            # Guardar como dict para compatibilidad con el resto del codigo
            self._view_images[name] = {"rgb": w_dep, "depth": w_dep}

            btn_mark = btn(f"Marcar zonas — {name.replace('_',' ')}", TEAL, TEAL_LIGHT)
            btn_mark.setFixedHeight(26)
            def _make_cb(vname):
                return lambda: self._open_zone_marker(vname)
            btn_mark.clicked.connect(_make_cb(name))
            cl.addWidget(btn_mark)

            grid.addWidget(cell, row, col)

            depth = STATE.depths.get(name)
            if depth is not None:
                valid = depth[(depth > 300) & (depth < 4000)]
                if len(valid) > 0:
                    d_norm = np.zeros_like(depth, dtype=np.uint8)
                    mask_v = (depth > 300) & (depth < 4000)
                    d_norm[mask_v] = ((depth[mask_v]-valid.min())/(valid.max()-valid.min())*255).astype(np.uint8)
                    w_dep.set_image_array(cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA))

        n = len(STATE.view_names)
        for i in range(n, 4):
            row, col = positions[i]
            grid.addWidget(QWidget(), row, col)

        self.img_layout.addWidget(grid_container, stretch=1)

    # ── MODIFICACIÓN 3 ─────────────────────────────────────────────────────────
    def _update_measurements_display(self):
        """
        Actualiza la tabla de medidas del panel izquierdo.

        STATE.measurements ahora es {zona: {"y_px", "circumference_cm", ...}}
        Se mantiene compatibilidad con el formato viejo {zona: float}.
        """
        for zone, lbl in self._meas_labels.items():
            data = STATE.measurements.get(zone)
            if isinstance(data, dict):
                val = data.get("circumference_cm")
            else:
                val = data          # compatibilidad con formato anterior
            lbl.setText(f"{val:.1f} cm" if val else "— cm")
        if STATE.height_cm > 0:
            self.lbl_height.setText(f"Altura estimada: {STATE.height_cm:.1f} cm")

    # ── MODIFICACIÓN 4 ─────────────────────────────────────────────────────────
    def _draw_measurement_lines(self):
        """
        Dibuja líneas horizontales de medición sobre la imagen frontal.

        Lee STATE.measurements[zona]["y_px"] (calculado por MediaPipe en
        extract_measurements) y llama img_w.set_lines() con las coordenadas
        normalizadas 0-1 que ImageWithOverlay espera.

        Si MediaPipe no estaba disponible y y_px viene del fallback manual,
        las líneas siguen apareciendo correctamente.

        Colores por zona:
            cuello  → rojo
            pecho   → naranja
            cintura → verde
            cadera  → azul
        """
        COLORS = {
            "cuello":  "#E24B4A",
            "pecho":   "#D85A30",
            "cintura": "#0F6E56",
            "cadera":  "#185FA5",
        }

        widgets = self._view_images.get("frontal")
        img_w = widgets["rgb"] if isinstance(widgets, dict) else widgets
        if img_w is None:
            return

        rgb = STATE.rgbs.get("frontal")
        if rgb is None:
            return

        H = rgb.shape[0]
        lines = []

        for zona, data in STATE.measurements.items():
            # Saltar zonas en formato viejo (float) que no tienen y_px
            if not isinstance(data, dict):
                continue
            y_px = data.get("y_px")
            cm   = data.get("circumference_cm")
            if y_px is None or cm is None:
                continue

            y_norm = y_px / H                   # normalizado 0-1
            color  = COLORS.get(zona, "#888780")
            label  = f"{zona}  {cm} cm"
            lines.append((y_norm, color, label))

        img_w.set_lines(lines)
    # ──────────────────────────────────────────────────────────────────────────

    def _open_zone_marker(self, view_name: str):
        import sys
        sys.path.insert(0, '.')
        from src.multi_view_pipeline import VIEW_ZONES, ZONES_PER_VIEW
        from src.preprocessing import preprocess_depth
        from src.segmentation  import segment_body

        rgb   = STATE.rgbs.get(view_name)
        depth = STATE.depths.get(view_name)
        if rgb is None or depth is None:
            QMessageBox.warning(self, "Sin imagen",
                f"Carga la imagen de {view_name} primero.")
            return

        roi = STATE.roi_params[view_name]
        depth_clean = preprocess_depth(depth)
        seg = segment_body(rgb, depth_clean,
            d_min_mm=roi["d_min"], d_max_mm=roi["d_max"],
            x_min=roi["x1"], x_max=roi["x2"],
            y_min=roi["y1"], y_max=roi["y2"])

        # Pasar depth como imagen de display — ZoneMarkerDialog usara el depth
        display_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # solo para compatibilidad

        dialog = ZoneMarkerDialog(
            view_name       = view_name,
            image_rgb       = display_rgb,
            zones_for_view  = ZONES_PER_VIEW[view_name],
            existing_zones  = dict(VIEW_ZONES.get(view_name, {})),
            parent          = self,
        )
        if dialog.exec_() == dialog.Accepted:
            VIEW_ZONES[view_name] = dialog.zone_rects
            self.lbl_status.setText(
                f"Zonas {view_name} guardadas ✓ ({len(dialog.zone_rects)} zonas)"
            )
            # Actualizar el panel depth de Page2 con las zonas marcadas
            img_w_dict = self._view_images.get(view_name)
            img_w = img_w_dict["depth"] if isinstance(img_w_dict, dict) else img_w_dict
            if img_w:
                depth = STATE.depths.get(view_name)
                if depth is not None:
                    valid = depth[(depth > 300) & (depth < 4000)]
                    d_norm = np.zeros_like(depth, dtype=np.uint8)
                    if len(valid) > 0:
                        mask_v = (depth > 300) & (depth < 4000)
                        d_norm[mask_v] = ((depth[mask_v]-valid.min())/(valid.max()-valid.min())*255).astype(np.uint8)
                    ann_bgr = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
                    COLORS_Z = {
                        "cuello":      (220, 60,  60),
                        "pecho":       (255, 144, 30),
                        "brazo_izq":   (200, 200, 0),
                        "brazo_der":   (160, 200, 0),
                        "cintura":     (0,   200, 0),
                        "cadera":      (0,   150, 200),
                        "muslo":       (200, 0,   200),
                        "rodilla":     (60,  60,  220),
                        "prof_pecho":  (255, 165, 0),
                        "prof_cintura":(200, 100, 0),
                    }
                    for z, r in dialog.zone_rects.items():
                        c = COLORS_Z.get(z, (180,180,180))
                        cv2.rectangle(ann_bgr, (r["x1"],r["y1"]),
                                      (r["x2"],r["y2"]), c, 2)
                        cv2.putText(ann_bgr, z, (r["x1"]+3, r["y1"]+13),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
                    img_w.set_image_array(ann_bgr)

    def _detect_landmarks_p2(self):
        """
        Botón 'Detectar landmarks' en Página 2.
        Corre MediaPipe sobre todas las vistas cargadas y actualiza
        los ImageWithOverlay de cada vista con RGB y depth anotados.
        """
        from src.pose_overlay import draw_landmarks_on_images

        if not STATE.rgbs:
            self.lbl_status.setText("Carga imágenes primero.")
            return

        self.btn_landmarks_p2.setEnabled(False)
        self.lbl_status.setText("Detectando landmarks...")
        QApplication.processEvents()

        found_any = False
        for name in STATE.view_names:
            rgb   = STATE.rgbs.get(name)
            depth = STATE.depths.get(name)
            if rgb is None:
                continue
            if depth is None:
                depth = np.zeros(rgb.shape[:2], dtype=np.float32)

            rgb_ann, depth_ann, ok = draw_landmarks_on_images(rgb, depth)

            if ok:
                widgets = self._view_images.get(name)
                if isinstance(widgets, dict):
                    # Nuevo formato — dict con claves "rgb" y "depth"
                    if rgb_ann is not None:
                        widgets["rgb"].set_image_array(rgb_ann)
                    if depth_ann is not None:
                        widgets["depth"].set_image_array(depth_ann)
                elif widgets is not None:
                    # Formato viejo — ImageWithOverlay directo (solo RGB)
                    if rgb_ann is not None:
                        widgets.set_image_array(rgb_ann)
                found_any = True

        self.btn_landmarks_p2.setEnabled(True)
        self.lbl_status.setText("Landmarks detectados ✓" if found_any
                                 else "No se detectó ninguna persona.")

    def _run_pipeline(self):
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Ejecutando pipeline...")
        self._thread = PipelineThread()
        self._thread.progress.connect(self.lbl_status.setText)
        self._thread.finished.connect(self._on_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_done(self, _):
        self.btn_run.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.lbl_status.setText("Pipeline completado ✓")
        self._update_measurements_display()

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Error en pipeline")
        QMessageBox.critical(self, "Error", msg[:800])


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
class Page3Results(QWidget):

    go_prev = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12,12,12,12)
        root.setSpacing(10)

        left = QWidget(); left.setFixedWidth(220)
        left.setStyleSheet(f"background:{BG_PANEL};border-radius:8px;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12,12,12,12); ll.setSpacing(6)

        ll.addWidget(section_label("Volumen por zona"))
        self._zone_cards = {}
        for zone in ["cuello","pecho","cintura","cadera","muslo","rodilla"]:
            card = QWidget()
            card.setStyleSheet(f"background:{BG_CARD};border-radius:6px;padding:4px;")
            cl = QVBoxLayout(card); cl.setContentsMargins(6,4,6,4); cl.setSpacing(2)
            lz = QLabel(zone.capitalize())
            lz.setStyleSheet(f"font-size:10px;color:{TEXT_SEC};")
            lv = QLabel("—")
            lv.setStyleSheet(f"font-size:13px;font-weight:500;color:{TEXT_PRI};")
            bar = QProgressBar(); bar.setRange(0,100); bar.setValue(50)
            bar.setTextVisible(False); bar.setFixedHeight(4)
            bar.setStyleSheet(f"""
                QProgressBar{{background:{BORDER};border-radius:2px;border:none;}}
                QProgressBar::chunk{{background:{CORAL};border-radius:2px;}}
            """)
            cl.addWidget(lz); cl.addWidget(lv); cl.addWidget(bar)
            self._zone_cards[zone] = {"val": lv, "bar": bar}
            ll.addWidget(card)

        ll.addWidget(hline())

        self.btn_run_smpl = btn("Comparar con SMPL", TEAL, TEAL_LIGHT)
        self.btn_run_smpl.clicked.connect(self._run_smpl)
        ll.addWidget(self.btn_run_smpl)

        self.btn_pdf = btn("Exportar PDF clínico", CORAL, CORAL_LIGHT)
        self.btn_pdf.clicked.connect(self._export_pdf)
        ll.addWidget(self.btn_pdf)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(f"font-size:11px;color:{TEXT_HINT};")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        ll.addStretch()

        self.btn_prev = btn("← Mediciones", TEXT_SEC, BG_CARD)
        self.btn_prev.clicked.connect(self.go_prev.emit)
        ll.addWidget(self.btn_prev)

        root.addWidget(left)

        center = QWidget()
        cl = QVBoxLayout(center); cl.setSpacing(8)

        cl.addWidget(section_label("Comparación de mallas — real vs referencia SMPL"))

        self.img_comparison = QLabel("Ejecuta la comparación SMPL para ver resultados")
        self.img_comparison.setAlignment(Qt.AlignCenter)
        self.img_comparison.setStyleSheet(
            f"background:{BG_CARD};border-radius:8px;color:{TEXT_HINT};font-size:12px;"
        )
        self.img_comparison.setMinimumHeight(300)
        cl.addWidget(self.img_comparison, stretch=2)

        cl.addWidget(hline())
        cl.addWidget(section_label("Tabla completa de medidas"))

        self.table_widget = QWidget()
        tl = QGridLayout(self.table_widget); tl.setSpacing(4)
        headers = ["Zona","Ancho frontal","Prof. lateral","Perímetro"]
        for ci, h in enumerate(headers):
            hl = QLabel(h)
            hl.setStyleSheet(
                f"background:{TEXT_PRI};color:white;padding:4px 8px;"
                f"font-size:11px;font-weight:500;border-radius:3px;"
            )
            tl.addWidget(hl, 0, ci)

        self._table_rows = {}
        for ri, zone in enumerate(ZONES, 1):
            bg = BG_CARD if ri % 2 == 0 else BG_PANEL
            row_widgets = []
            for ci in range(4):
                lbl = QLabel("—")
                lbl.setStyleSheet(
                    f"background:{bg};padding:4px 8px;font-size:11px;color:{TEXT_PRI};"
                )
                tl.addWidget(lbl, ri, ci)
                row_widgets.append(lbl)
            self._table_rows[zone] = row_widgets

        cl.addWidget(self.table_widget)
        root.addWidget(center, stretch=1)

    def showEvent(self, e):
        super().showEvent(e)
        self._update_table()

    def _update_table(self):
        """
        Actualiza la tabla de Page3.
        Lee circumference_cm del dict completo si está disponible.
        """
        for zone, row in self._table_rows.items():
            row[0].setText(zone.capitalize())
            d = STATE.diag.get(zone, {})
            row[1].setText(f"{d.get('w_front_cm','—')} cm" if d.get('w_front_cm') else "—")
            row[2].setText(f"{d.get('w_side_cm','—')} cm"  if d.get('w_side_cm')  else "—")
            # Leer circumference_cm del dict completo o del formato viejo
            meas_data = STATE.measurements.get(zone)
            if isinstance(meas_data, dict):
                val = meas_data.get("circumference_cm")
            else:
                val = meas_data
            row[3].setText(f"{val:.1f} cm" if val else "—")

    def _run_smpl(self):
        self.lbl_status.setText("Ajustando SMPL (~40s primera vez)...")
        self.btn_run_smpl.setEnabled(False)
        from PyQt5.QtCore import QThread, pyqtSignal as Signal
        class SmplThread(QThread):
            done  = Signal(str)
            error = Signal(str)
            def run(self):
                try:
                    from src.smpl_fitting     import (load_smpl_model,
                                                       generate_smpl_mesh,
                                                       get_vertices)
                    from src.regression_model import UserInputs, predict_measurements
                    from src.volume_comparison import (create_comparison_mesh,
                                                        save_comparison_figure,
                                                        compute_zone_statistics)
                    from src.smpl_cache        import (load_cached_betas,
                                                        save_cached_betas)

                    p = STATE.patient
                    # Usar género correcto del paciente para SMPL
                    smpl_gender = "male" if p["sex"] == "male" else "female"
                    smpl_model = load_smpl_model("models", smpl_gender)

                    # ── Malla REAL: medidas del paciente con sus parametros actuales ──
                    real_inputs = UserInputs(
                        body_fat = p["body_fat"],
                        sex      = p["sex"],
                        age      = p["age"],
                        weight   = p["weight"],
                        height   = p["height"],
                    )
                    real_meas = predict_measurements(real_inputs)
                    real_fmt  = {
                        "cuello":  {"circumference_cm": real_meas["neck"]},
                        "pecho":   {"circumference_cm": real_meas["chest"]},
                        "cintura": {"circumference_cm": real_meas["abdomen"]},
                        "cadera":  {"circumference_cm": real_meas["hip"]},
                    }
                    real_target = {z: round(float(d["circumference_cm"]),1)
                                   for z, d in real_fmt.items()}
                    cached_real = load_cached_betas(real_target, smpl_gender)
                    if cached_real is not None:
                        verts_real = get_vertices(smpl_model, cached_real)
                    else:
                        verts_real, _, betas_real_opt = generate_smpl_mesh(
                            real_fmt, "models", smpl_gender
                        )
                        save_cached_betas(real_target, betas_real_opt, smpl_gender)

                    # ── Malla SINTETICA: mismo paciente con peso ideal (IMC 22) ──
                    # peso_ideal = 22 * altura^2
                    ideal_weight = round(22.0 * p["height"] ** 2, 1)
                    synth_inputs = UserInputs(
                        body_fat = p["body_fat"],
                        sex      = p["sex"],
                        age      = p["age"],
                        weight   = ideal_weight,
                        height   = p["height"],
                    )
                    synth_meas = predict_measurements(synth_inputs)
                    synth_fmt  = {
                        "cuello":  {"circumference_cm": synth_meas["neck"]},
                        "pecho":   {"circumference_cm": synth_meas["chest"]},
                        "cintura": {"circumference_cm": synth_meas["abdomen"]},
                        "cadera":  {"circumference_cm": synth_meas["hip"]},
                    }
                    target_flat = {z: round(float(d["circumference_cm"]),1)
                                   for z, d in synth_fmt.items()}
                    cached_synth = load_cached_betas(target_flat, smpl_gender)
                    if cached_synth is not None:
                        verts_synth = get_vertices(smpl_model, cached_synth)
                    else:
                        verts_synth, _, betas_synth = generate_smpl_mesh(
                            synth_fmt, "models", smpl_gender
                        )
                        save_cached_betas(target_flat, betas_synth, smpl_gender)

                    print(f"  Peso actual: {p['weight']}kg  →  Peso ideal IMC22: {ideal_weight}kg")
                    print(f"  Real — cintura: {real_meas['abdomen']:.1f}cm  cadera: {real_meas['hip']:.1f}cm")
                    print(f"  Ideal — cintura: {synth_meas['abdomen']:.1f}cm  cadera: {synth_meas['hip']:.1f}cm")

                    _, _, signed_dists = create_comparison_mesh(
                        verts_real, verts_synth, smpl_model.faces
                    )
                    save_comparison_figure(
                        verts_real, verts_synth, signed_dists,
                        "output/volume_comparison.png"
                    )
                    zone_stats = compute_zone_statistics(
                        verts_real, verts_synth, signed_dists
                    )
                    STATE.smpl_result = {
                        "zone_stats":   zone_stats,
                        "img_path":     "output/volume_comparison.png",
                        "real_meas":    real_meas,
                        "synth_meas":   synth_meas,
                        "ideal_weight": ideal_weight,
                    }
                    self.done.emit("output/volume_comparison.png")
                except Exception:
                    import traceback
                    self.error.emit(traceback.format_exc())

        self._smpl_thr = SmplThread()
        self._smpl_thr.done.connect(self._on_smpl_done)
        self._smpl_thr.error.connect(self._on_smpl_error)
        self._smpl_thr.start()

    def _on_smpl_done(self, img_path):
        self.btn_run_smpl.setEnabled(True)
        ideal_w = (STATE.smpl_result or {}).get("ideal_weight", "—")
        self.lbl_status.setText(
            f"Completado ✓  |  Peso ideal IMC22: {ideal_w} kg"
        )
        if Path(img_path).exists():
            pix = QPixmap(img_path).scaled(
                self.img_comparison.width(),
                self.img_comparison.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.img_comparison.setPixmap(pix)
        if STATE.smpl_result:
            for zone, card in self._zone_cards.items():
                stats = STATE.smpl_result["zone_stats"].get(zone,{})
                mean  = stats.get("mean_cm", 0)
                pct   = stats.get("pct_excess", 50)
                sign  = "+" if mean >= 0 else ""
                card["val"].setText(f"{sign}{mean:.2f} cm")
                card["bar"].setValue(int(pct))

    def _on_smpl_error(self, msg):
        self.btn_run_smpl.setEnabled(True)
        self.lbl_status.setText("Error SMPL")
        QMessageBox.critical(self, "Error SMPL", msg[:600])

    def _export_pdf(self):
        if not STATE.pipeline_done:
            QMessageBox.warning(self,"Sin datos","Ejecuta el pipeline primero.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar PDF",
            str(Path.home() / "Desktop" / "reporte_corporal.pdf"),
            "PDF (*.pdf)"
        )
        if not save_path: return
        try:
            from src.pdf_report import generate_pdf

            class _Card:
                def __init__(self, v):
                    self._v = v
                    self.lbl_val = self
                def text(self):
                    s = "+" if self._v >= 0 else ""
                    return f"{s}{self._v:.2f} cm"

            zone_cards_simple = {}
            if STATE.smpl_result:
                for z, stats in STATE.smpl_result["zone_stats"].items():
                    zone_cards_simple[z] = _Card(stats.get("mean_cm",0))

            p = STATE.patient
            # Extraer circumference_cm del dict completo para el PDF
            meas_flat = {}
            for z, d in STATE.measurements.items():
                if isinstance(d, dict):
                    meas_flat[z] = d.get("circumference_cm")
                else:
                    meas_flat[z] = d

            generate_pdf(
                img_path    = (STATE.smpl_result or {}).get(
                                  "img_path","output/volume_comparison.png"),
                zone_cards  = zone_cards_simple,
                output_path = save_path,
                patient_data= {
                    "name":     p["name"],
                    "sex":      "Femenino" if p["sex"]=="female" else "Masculino",
                    "age":      p["age"],
                    "weight":   p["weight"],
                    "height":   p["height"],
                    "body_fat": p["body_fat"],
                },
                meas_real  = meas_flat,
                synth_meas = (STATE.smpl_result or {}).get("synth_meas"),
            )
            self.lbl_status.setText("PDF exportado ✓")
            QMessageBox.information(self,"PDF","Guardado en:\n"+save_path)
        except Exception:
            import traceback
            QMessageBox.critical(self,"Error PDF",traceback.format_exc()[:800])


# ══════════════════════════════════════════════════════════════════════════════
# VENTANA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Body3D Reconstruction")
        self.setMinimumSize(1280, 800)
        self._build()
        self._apply_styles()

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        nav_bar = QWidget()
        nav_bar.setFixedHeight(48)
        nav_bar.setStyleSheet(f"background:{BG_PANEL};border-bottom:1px solid {BORDER};")
        nb = QHBoxLayout(nav_bar)
        nb.setContentsMargins(16,0,16,0); nb.setSpacing(0)

        title = QLabel("Body3D Reconstruction")
        title.setStyleSheet(
            f"font-size:14px;font-weight:500;color:{TEXT_PRI};margin-right:32px;"
        )
        nb.addWidget(title)

        self._tab_btns = []
        for i, (label, color) in enumerate([
            ("1 · Captura", TEAL),
            ("2 · Mediciones", PURPLE),
            ("3 · Resultados", CORAL),
        ]):
            tb = QPushButton(label)
            tb.setCheckable(True)
            tb.setFixedHeight(48)
            tb.setStyleSheet(f"""
                QPushButton{{background:transparent;color:{TEXT_SEC};
                    border:none;border-bottom:3px solid transparent;
                    padding:0 20px;font-size:12px;}}
                QPushButton:checked{{color:{color};
                    border-bottom:3px solid {color};font-weight:500;}}
                QPushButton:hover:!checked{{color:{TEXT_PRI};}}
            """)
            tb.clicked.connect(lambda _, idx=i: self._go_to(idx))
            nb.addWidget(tb)
            self._tab_btns.append(tb)

        nb.addStretch()
        root.addWidget(nav_bar)

        self.stack = QStackedWidget()
        self.p1 = Page1Capture()
        self.p2 = Page2Measurements()
        self.p3 = Page3Results()
        self.stack.addWidget(self.p1)
        self.stack.addWidget(self.p2)
        self.stack.addWidget(self.p3)
        root.addWidget(self.stack, stretch=1)

        self.p1.go_next.connect(lambda: self._go_to(1))
        self.p2.go_next.connect(lambda: self._go_to(2))
        self.p2.go_prev.connect(lambda: self._go_to(0))
        self.p3.go_prev.connect(lambda: self._go_to(1))

        self._go_to(0)

    def _go_to(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, tb in enumerate(self._tab_btns):
            tb.setChecked(i == idx)

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow{{background:{BG_MAIN};}}
            QSlider::groove:horizontal{{height:4px;background:{BORDER};border-radius:2px;}}
            QSlider::handle:horizontal{{width:14px;height:14px;margin:-5px 0;
                border-radius:7px;background:{TEAL};}}
            QSlider::sub-page:horizontal{{background:{TEAL};border-radius:2px;}}
            QComboBox{{background:{BG_CARD};border:0.5px solid {BORDER};
                border-radius:6px;padding:4px 8px;font-size:12px;color:{TEXT_PRI};}}
            QScrollArea{{border:none;}}
            QGroupBox{{font-size:12px;}}
        """)