"""
main_window.py
--------------
Ventana principal Body3D Reconstruction.
Mejoras:
    - Cache SMPL corregido (hash consistente a 1 decimal)
    - Importar .npy desde disco con conversión automática
    - PDF sin fecha debajo del título
    - Navegación de imagen con botones zoom +/- y reset
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QLineEdit, QComboBox, QGridLayout, QVBoxLayout,
    QHBoxLayout, QSizePolicy, QFrame, QProgressBar, QMessageBox,
    QFileDialog, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


# ── Colores ───────────────────────────────────────────────────────────────────
TEAL       = "#0F6E56"
TEAL_LIGHT = "#E1F5EE"
CORAL      = "#993C1D"
CORAL_LIGHT= "#FAECE7"
BLUE       = "#378ADD"
BG_MAIN    = "#F5F5F3"
BG_PANEL   = "#FFFFFF"
BG_CARD    = "#F1EFE8"
BORDER     = "#D3D1C7"
TEXT_PRI   = "#2C2C2A"
TEXT_SEC   = "#5F5E5A"
TEXT_HINT  = "#888780"


# ── Widget de imagen con zoom y navegación ────────────────────────────────────
class ZoomableImage(QWidget):
    """
    Imagen con zoom (+/-/reset) y pan (arrastrar).
    Incluye barra de herramientas con botones visibles.
    """

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Barra de herramientas de zoom
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)

        self.btn_zoom_in  = self._tool_btn("＋  Zoom +")
        self.btn_zoom_out = self._tool_btn("－  Zoom −")
        self.btn_reset    = self._tool_btn("⊡  Reset")

        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_reset.clicked.connect(self._zoom_reset)

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setStyleSheet(
            f"font-size:11px; color:{TEXT_SEC}; min-width:36px;"
        )

        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.btn_reset)
        toolbar.addWidget(self.lbl_zoom)
        toolbar.addStretch()
        toolbar.addWidget(QLabel(
            "Rueda = zoom  ·  Arrastrar = mover  ·  Doble clic = reset"
        ))
        layout.addLayout(toolbar)

        # Área de scroll que contiene la imagen
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet(
            f"background:{BG_PANEL}; border-radius:8px;"
            f" border:0.5px solid {BORDER};"
        )
        self.scroll.setMinimumHeight(300)

        self.lbl_img = QLabel("Ejecuta el pipeline para ver la comparación")
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setStyleSheet(
            f"background:{BG_PANEL}; color:{TEXT_HINT}; font-size:12px;"
        )
        self.lbl_img.setMinimumHeight(300)
        self.scroll.setWidget(self.lbl_img)

        layout.addWidget(self.scroll)

        self._pixmap_orig = None
        self._scale       = 1.0
        self._drag_pos    = None

        # Habilitar eventos de ratón en la imagen
        self.lbl_img.setMouseTracking(True)
        self.lbl_img.mousePressEvent   = self._mouse_press
        self.lbl_img.mouseMoveEvent    = self._mouse_move
        self.lbl_img.mouseReleaseEvent = self._mouse_release
        self.lbl_img.mouseDoubleClickEvent = lambda e: self._zoom_reset()
        self.lbl_img.wheelEvent        = self._wheel

    def _tool_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background:{BG_CARD}; color:{TEXT_PRI};
                border:0.5px solid {BORDER}; border-radius:5px;
                padding:3px 10px; font-size:11px;
            }}
            QPushButton:hover {{
                background:{BORDER};
            }}
        """)
        btn.setFixedHeight(24)
        return btn

    def set_image(self, path: str):
        self._pixmap_orig = QPixmap(path)
        self._scale       = 1.0
        self._refresh()

    def set_text(self, text: str):
        self._pixmap_orig = None
        self.lbl_img.setPixmap(QPixmap())
        self.lbl_img.setText(text)

    def _refresh(self):
        if self._pixmap_orig is None:
            return
        w = int(self._pixmap_orig.width()  * self._scale)
        h = int(self._pixmap_orig.height() * self._scale)
        pix = self._pixmap_orig.scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_img.setPixmap(pix)
        self.lbl_img.resize(w, h)
        self.lbl_zoom.setText(f"{int(self._scale * 100)}%")

    def _zoom_in(self):
        self._scale = min(self._scale * 1.25, 10.0)
        self._refresh()

    def _zoom_out(self):
        self._scale = max(self._scale * 0.8, 0.1)
        self._refresh()

    def _zoom_reset(self):
        self._scale = 1.0
        self._refresh()

    def _wheel(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 0.87
        self._scale = max(0.1, min(self._scale * factor, 10.0))
        self._refresh()
        e.accept()

    def _mouse_press(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.pos()
            self.lbl_img.setCursor(Qt.ClosedHandCursor)

    def _mouse_move(self, e):
        if self._drag_pos is not None:
            delta = e.pos() - self._drag_pos
            self._drag_pos = e.pos()
            sb_h = self.scroll.horizontalScrollBar()
            sb_v = self.scroll.verticalScrollBar()
            sb_h.setValue(sb_h.value() - delta.x())
            sb_v.setValue(sb_v.value() - delta.y())

    def _mouse_release(self, e):
        self._drag_pos = None
        self.lbl_img.setCursor(Qt.ArrowCursor)


# ── Hilo del pipeline ─────────────────────────────────────────────────────────
class PipelineThread(QThread):
    """
    Ejecuta Steps 8 y 9 en hilo separado.
    Usa cache SMPL para evitar reoptimizar si los parámetros no cambian.
    """
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, rgb: np.ndarray, depth: np.ndarray,
                 user_params: dict):
        super().__init__()
        self.rgb         = rgb
        self.depth       = depth
        self.user_params = user_params

    def run(self):
        try:
            from src.smpl_fitting      import (load_smpl_model, get_vertices,
                                                get_all_measurements,
                                                generate_smpl_mesh)
            from src.regression_model  import UserInputs, predict_measurements
            from src.volume_comparison import (create_comparison_mesh,
                                               save_comparison_figure,
                                               compute_zone_statistics)
            from src.smpl_cache        import (load_cached_betas,
                                               save_cached_betas)

            # ── Malla SMPL real ───────────────────────────────────────────────
            self.progress.emit("Cargando modelo SMPL real...")
            smpl_model    = load_smpl_model("models", "neutral")
            betas_persona = np.array([-4.3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            verts_real    = get_vertices(smpl_model, betas_persona)
            meas_real     = get_all_measurements(smpl_model, betas_persona)

            # ── Malla sintética con cache ─────────────────────────────────────
            self.progress.emit("Preparando referencia sintética...")
            p = self.user_params
            user_inputs = UserInputs(
                body_fat = p["body_fat"],
                sex      = p["sex"],
                age      = p["age"],
                weight   = p["weight"],
                height   = p["height"],
            )
            synth_meas = predict_measurements(user_inputs)
            synth_fmt  = {
                "cuello":  {"circumference_cm": synth_meas["neck"]},
                "pecho":   {"circumference_cm": synth_meas["chest"]},
                "cintura": {"circumference_cm": synth_meas["abdomen"]},
                "cadera":  {"circumference_cm": synth_meas["hip"]},
            }

            # Target plano con redondeo a 1 decimal para cache consistente
            target_flat = {
                z: round(float(d["circumference_cm"]), 1)
                for z, d in synth_fmt.items()
            }

            cached_betas = load_cached_betas(target_flat)

            if cached_betas is not None:
                self.progress.emit("Usando referencia cacheada (<1s)...")
                verts_synth = get_vertices(smpl_model, cached_betas)
            else:
                self.progress.emit(
                    "Generando referencia sintética (~40s primera vez)..."
                )
                verts_synth, _, betas_synth = generate_smpl_mesh(
                    synth_fmt, "models", "neutral"
                )
                save_cached_betas(target_flat, betas_synth)

            # ── Comparación volumétrica ───────────────────────────────────────
            self.progress.emit("Calculando diferencia volumétrica...")
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

            self.finished.emit({
                "meas_real":   meas_real,
                "synth_meas":  synth_meas,
                "zone_stats":  zone_stats,
                "img_path":    "output/volume_comparison.png",
                "user_params": p,
            })

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ── Separadores ───────────────────────────────────────────────────────────────
def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


# ── Tarjeta de zona ───────────────────────────────────────────────────────────
class ZoneCard(QWidget):
    """Muestra zona, diferencia en cm y barra de color."""

    def __init__(self, zone: str, value_cm: float = 0.0,
                 pct_excess: float = 50.0):
        super().__init__()
        self.setStyleSheet(f"ZoneCard {{ background:{BG_CARD}; border-radius:8px; }}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)

        self.lbl_zone = QLabel(zone.capitalize())
        self.lbl_zone.setStyleSheet(f"font-size:11px; color:{TEXT_SEC};")

        sign = "+" if value_cm >= 0 else ""
        self.lbl_val = QLabel(f"{sign}{value_cm:.2f} cm")
        self.lbl_val.setStyleSheet(
            f"font-size:14px; font-weight:500; color:{TEXT_PRI};"
        )

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(int(pct_excess))
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        self._set_bar_color(value_cm)

        layout.addWidget(self.lbl_zone)
        layout.addWidget(self.lbl_val)
        layout.addWidget(self.bar)

    def _set_bar_color(self, value_cm: float):
        color = CORAL if value_cm >= 0 else BLUE
        self.bar.setStyleSheet(f"""
            QProgressBar {{ background:{BORDER}; border-radius:2px; border:none; }}
            QProgressBar::chunk {{ background:{color}; border-radius:2px; }}
        """)

    def update_values(self, value_cm: float, pct_excess: float):
        sign = "+" if value_cm >= 0 else ""
        self.lbl_val.setText(f"{sign}{value_cm:.2f} cm")
        self.bar.setValue(int(pct_excess))
        self._set_bar_color(value_cm)


# ── Ventana principal ─────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Body3D Reconstruction")
        self.setMinimumSize(1200, 750)

        self.cam           = None
        self.timer         = QTimer()
        self.pipeline_thr  = None
        self.last_rgb      = None
        self.last_depth    = None
        self._last_results = None
        self._zone_values  = {}

        self._build_ui()
        self._apply_styles()

    # ── Construcción de la UI ─────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_left_panel(),  stretch=0)
        root.addWidget(self._vline(),              stretch=0)
        root.addWidget(self._build_center_panel(), stretch=1)
        root.addWidget(self._vline(),              stretch=0)
        root.addWidget(self._build_right_panel(),  stretch=0)

    def _vline(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFixedWidth(1)
        f.setStyleSheet(f"color: {BORDER};")
        return f

    # ── Panel izquierdo ───────────────────────────────────────────────────────
    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(240)
        w.setStyleSheet(f"background:{BG_PANEL};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        lay.addWidget(self._section_label("Cámara"))

        self.lbl_rgb = QLabel("RGB — sin señal")
        self.lbl_rgb.setAlignment(Qt.AlignCenter)
        self.lbl_rgb.setFixedHeight(95)
        self.lbl_rgb.setStyleSheet(
            f"background:{BG_CARD}; border-radius:6px;"
            f" color:{TEXT_HINT}; font-size:11px;"
        )
        lay.addWidget(self.lbl_rgb)

        self.lbl_depth = QLabel("Depth — sin señal")
        self.lbl_depth.setAlignment(Qt.AlignCenter)
        self.lbl_depth.setFixedHeight(95)
        self.lbl_depth.setStyleSheet(
            f"background:{BG_CARD}; border-radius:6px;"
            f" color:{TEXT_HINT}; font-size:11px;"
        )
        lay.addWidget(self.lbl_depth)

        btn_row = QHBoxLayout()
        self.btn_start = self._btn("▶ Iniciar",  TEAL,  TEAL_LIGHT)
        self.btn_stop  = self._btn("■ Detener",  CORAL, CORAL_LIGHT)
        self.btn_start.clicked.connect(self._start_camera)
        self.btn_stop.clicked.connect(self._stop_camera)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        lay.addLayout(btn_row)

        self.btn_capture = self._btn("⬤ Capturar frame", TEAL, TEAL_LIGHT)
        self.btn_capture.clicked.connect(self._capture_frame)
        self.btn_capture.setEnabled(False)
        lay.addWidget(self.btn_capture)

        self.btn_import = self._btn("📂 Importar imágenes", "#444441", BG_CARD)
        self.btn_import.clicked.connect(self._import_images)
        lay.addWidget(self.btn_import)

        lay.addWidget(hline())

        lay.addWidget(self._section_label("Parámetros del usuario"))

        bf_lbl = QLabel("% grasa corporal")
        bf_lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SEC};")
        lay.addWidget(bf_lbl)

        slider_row = QHBoxLayout()
        self.slider_bf = QSlider(Qt.Horizontal)
        self.slider_bf.setRange(5, 45)
        self.slider_bf.setValue(15)
        self.lbl_bf_val = QLabel("15%")
        self.lbl_bf_val.setFixedWidth(32)
        self.lbl_bf_val.setStyleSheet(
            f"font-size:12px; font-weight:500; color:{TEXT_PRI};"
        )
        self.slider_bf.valueChanged.connect(
            lambda v: self.lbl_bf_val.setText(f"{v}%")
        )
        slider_row.addWidget(self.slider_bf)
        slider_row.addWidget(self.lbl_bf_val)
        lay.addLayout(slider_row)

        row1 = QHBoxLayout()
        self.cmb_sex = QComboBox()
        self.cmb_sex.addItems(["Masculino", "Femenino"])
        self.cmb_sex.setStyleSheet(self._input_style())
        self.inp_age = QLineEdit("30")
        self.inp_age.setStyleSheet(self._input_style())
        row1.addWidget(self._labeled("Sexo", self.cmb_sex))
        row1.addWidget(self._labeled("Edad", self.inp_age))
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.inp_weight = QLineEdit("80.0")
        self.inp_weight.setStyleSheet(self._input_style())
        self.inp_height = QLineEdit("1.75")
        self.inp_height.setStyleSheet(self._input_style())
        row2.addWidget(self._labeled("Peso (kg)", self.inp_weight))
        row2.addWidget(self._labeled("Talla (m)", self.inp_height))
        lay.addLayout(row2)

        self.inp_name = QLineEdit("")
        self.inp_name.setPlaceholderText("Nombre del paciente")
        self.inp_name.setStyleSheet(self._input_style())
        lay.addWidget(self._labeled("Nombre (para PDF)", self.inp_name))

        lay.addWidget(hline())

        self.btn_run = self._btn("⚡ Ejecutar pipeline", TEAL, TEAL_LIGHT)
        self.btn_run.clicked.connect(self._run_pipeline)
        lay.addWidget(self.btn_run)

        self.btn_pdf = self._btn("⬇ Exportar PDF clínico", "#444441", BG_CARD)
        self.btn_pdf.clicked.connect(self._export_pdf)
        self.btn_pdf.setEnabled(False)
        lay.addWidget(self.btn_pdf)

        self.lbl_status = QLabel("En espera")
        self.lbl_status.setStyleSheet(
            f"font-size:11px; color:{TEXT_HINT}; margin-top:4px;"
        )
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        lay.addStretch()
        return w

    # ── Panel central ─────────────────────────────────────────────────────────
    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{BG_MAIN};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        lay.addWidget(self._section_label(
            "Comparación volumétrica — real vs referencia sintética"
        ))

        self.img_comparison = ZoomableImage()
        lay.addWidget(self.img_comparison, stretch=2)

        lay.addWidget(hline())

        lay.addWidget(self._section_label(
            "Medidas antropométricas — malla real"
        ))

        self.meas_grid  = QGridLayout()
        self.meas_grid.setSpacing(6)
        self.meas_cards = {}

        for i, z in enumerate(["Cuello", "Pecho", "Cintura", "Cadera"]):
            card, lbl = self._meas_card(z, "— cm")
            self.meas_cards[z.lower()] = lbl
            self.meas_grid.addWidget(card, 0, i)

        for i, z in enumerate(["Muslo", "Rodilla", "Muñeca"]):
            card, lbl = self._meas_card(z, "— cm")
            self.meas_cards[z.lower()] = lbl
            self.meas_grid.addWidget(card, 1, i)

        lay.addLayout(self.meas_grid)
        return w

    # ── Panel derecho ─────────────────────────────────────────────────────────
    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(200)
        w.setStyleSheet(f"background:{BG_PANEL};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)

        lay.addWidget(self._section_label("Volumen por zona"))

        self.zone_cards = {}
        for z in ["cabeza", "cuello", "pecho", "cintura",
                  "cadera", "muslos", "piernas"]:
            card = ZoneCard(z)
            self.zone_cards[z] = card
            lay.addWidget(card)

        lay.addWidget(hline())

        for color, label in [(CORAL, "exceso de volumen"),
                              (BLUE,  "déficit de volumen")]:
            row = QHBoxLayout()
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background:{color}; border-radius:2px;")
            lbl = QLabel(label)
            lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SEC};")
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            lay.addLayout(row)

        lay.addStretch()
        return w

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size:11px; font-weight:500; color:{TEXT_SEC};"
        )
        return lbl

    def _btn(self, text: str, fg: str, bg: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background:{bg}; color:{fg};
                border:0.5px solid {fg}; border-radius:6px;
                padding:6px 10px; font-size:12px;
            }}
            QPushButton:disabled {{
                background:{BG_CARD}; color:{TEXT_HINT};
                border-color:{BORDER};
            }}
        """)
        return btn

    def _input_style(self) -> str:
        return (
            f"background:{BG_CARD}; border:0.5px solid {BORDER};"
            f" border-radius:6px; padding:4px 8px; font-size:12px;"
            f" color:{TEXT_PRI};"
        )

    def _labeled(self, label: str, widget: QWidget) -> QWidget:
        c = QWidget()
        l = QVBoxLayout(c)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)
        lb = QLabel(label)
        lb.setStyleSheet(f"font-size:11px; color:{TEXT_SEC};")
        l.addWidget(lb)
        l.addWidget(widget)
        return c

    def _meas_card(self, zone: str,
                   value: str) -> tuple[QWidget, QLabel]:
        card = QWidget()
        card.setStyleSheet(
            f"background:{BG_PANEL}; border:0.5px solid {BORDER};"
            f" border-radius:6px;"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        lbl_z = QLabel(zone)
        lbl_z.setStyleSheet(f"font-size:10px; color:{TEXT_SEC};")
        lbl_v = QLabel(value)
        lbl_v.setStyleSheet(
            f"font-size:13px; font-weight:500; color:{TEXT_PRI};"
        )
        lay.addWidget(lbl_z)
        lay.addWidget(lbl_v)
        return card, lbl_v

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background:{BG_MAIN}; }}
            QSlider::groove:horizontal {{
                height:4px; background:{BORDER}; border-radius:2px;
            }}
            QSlider::handle:horizontal {{
                width:14px; height:14px; margin:-5px 0;
                border-radius:7px; background:{TEAL};
            }}
            QSlider::sub-page:horizontal {{
                background:{TEAL}; border-radius:2px;
            }}
            QComboBox {{
                background:{BG_CARD}; border:0.5px solid {BORDER};
                border-radius:6px; padding:4px 8px;
                font-size:12px; color:{TEXT_PRI};
            }}
            QScrollArea {{
                border:none;
            }}
        """)

    # ── Cámara ────────────────────────────────────────────────────────────────
    def _start_camera(self):
        from src.camera import RealSenseCamera, CameraConfig
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            if len(ctx.devices) == 0:
                raise RuntimeError("Sin dispositivo")
            self.cam = RealSenseCamera(simulate=False, config=CameraConfig())
            mode_txt = "RealSense D455"
        except Exception:
            self.cam = RealSenseCamera(simulate=True, config=CameraConfig())
            mode_txt = "Simulación"

        self.cam.start()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(33)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_capture.setEnabled(True)
        self.lbl_status.setText(f"Cámara activa — {mode_txt}")

    def _stop_camera(self):
        self.timer.stop()
        if self.cam:
            self.cam.stop()
            self.cam = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.lbl_rgb.setText("RGB — sin señal")
        self.lbl_depth.setText("Depth — sin señal")
        self.lbl_status.setText("Cámara detenida")

    def _update_frame(self):
        if not self.cam:
            return
        try:
            rgb, depth      = self.cam.get_frames()
            self.last_rgb   = rgb
            self.last_depth = depth
            self._show_frame(self.lbl_rgb, rgb)
            self._show_frame(self.lbl_depth,
                             self.cam.get_depth_colormap(depth))
        except Exception as e:
            self.lbl_status.setText(f"Error cámara: {e}")
            self._stop_camera()

    def _show_frame(self, label: QLabel, frame: np.ndarray):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg    = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        pix     = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    def _capture_frame(self):
        if self.last_rgb is None:
            self.lbl_status.setText("No hay frame disponible")
            return
        os.makedirs("data/captured", exist_ok=True)
        cv2.imwrite("data/captured/rgb.png", self.last_rgb)
        if self.last_depth is not None:
            np.save("data/captured/depth.npy", self.last_depth)
        self.lbl_status.setText("Frame capturado ✓")

    def _import_images(self):
        """Importar RGB (.png/.jpg) y Depth (.npy) desde disco."""
        # Seleccionar RGB
        rgb_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen RGB",
            str(Path.home()),
            "Imágenes (*.png *.jpg *.jpeg);;Todas (*.*)"
        )
        if not rgb_path:
            return

        # Seleccionar depth
        depth_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar mapa de profundidad (.npy)",
            str(Path(rgb_path).parent),
            "NumPy (*.npy);;Todas (*.*)"
        )

        try:
            # Cargar RGB
            self.last_rgb = cv2.imread(rgb_path)
            if self.last_rgb is None:
                raise ValueError(f"No se pudo leer: {rgb_path}")
            self._show_frame(self.lbl_rgb, self.last_rgb)

            # Cargar y visualizar depth
            if depth_path:
                depth_raw = np.load(depth_path)
                if depth_raw.dtype == np.uint16:
                    self.last_depth = depth_raw.astype(np.float32) * 0.001
                else:
                    self.last_depth = depth_raw.astype(np.float32)
                self.last_depth[
                    (self.last_depth < 0.1) | (self.last_depth > 6.0)
                ] = 0.0

                # Visualizar depth como colormap JET
                valid = self.last_depth[self.last_depth > 0]
                if valid.size > 0:
                    d_norm = np.zeros_like(self.last_depth)
                    d_norm[self.last_depth > 0] = (
                        (self.last_depth[self.last_depth > 0] - valid.min())
                        / (valid.max() - valid.min()) * 255
                    )
                    depth_vis = cv2.applyColorMap(
                        d_norm.astype(np.uint8), cv2.COLORMAP_JET
                    )
                    self._show_frame(self.lbl_depth, depth_vis)

            depth_info = f" + {Path(depth_path).name}" if depth_path else ""
            self.lbl_status.setText(
                f"Importado ✓  {Path(rgb_path).name}{depth_info}"
            )

        except Exception as e:
            QMessageBox.warning(self, "Error al importar", str(e))
            
    # ── Pipeline ──────────────────────────────────────────────────────────────
    def _get_user_params(self) -> dict | None:
        try:
            params = {
                "body_fat": float(self.slider_bf.value()),
                "sex":      "male" if self.cmb_sex.currentIndex() == 0
                            else "female",
                "age":      float(self.inp_age.text()),
                "weight":   float(self.inp_weight.text()),
                "height":   float(self.inp_height.text()),
                "name":     self.inp_name.text().strip()
                            or "No especificado",
            }
            assert 5   <= params["body_fat"] <= 45,  "% grasa fuera de rango (5–45)"
            assert 10  <= params["age"]      <= 100, "Edad inválida (10–100)"
            assert 20  <= params["weight"]   <= 300, "Peso inválido (20–300 kg)"
            assert 1.0 <= params["height"]   <= 2.5, "Talla inválida (1.0–2.5 m)"
            return params
        except (ValueError, AssertionError) as e:
            QMessageBox.warning(self, "Parámetros inválidos", str(e))
            return None

    def _run_pipeline(self):
        params = self._get_user_params()
        if params is None:
            return

        rgb   = self.last_rgb   if self.last_rgb   is not None \
                else np.zeros((720, 1280, 3),  dtype=np.uint8)
        depth = self.last_depth if self.last_depth is not None \
                else np.zeros((720, 1280),     dtype=np.float32)

        self.btn_run.setEnabled(False)
        self.btn_pdf.setEnabled(False)
        self.img_comparison.set_text("Procesando...")
        self.lbl_status.setText("Iniciando pipeline...")

        self.pipeline_thr = PipelineThread(rgb, depth, params)
        self.pipeline_thr.progress.connect(self.lbl_status.setText)
        self.pipeline_thr.finished.connect(self._on_pipeline_done)
        self.pipeline_thr.error.connect(self._on_pipeline_error)
        self.pipeline_thr.start()

    def _on_pipeline_done(self, results: dict):
        self.btn_run.setEnabled(True)
        self.btn_pdf.setEnabled(True)
        self._last_results = results
        self.lbl_status.setText(
            "Pipeline completado ✓  —  puedes modificar parámetros y volver a ejecutar"
        )

        img_path = results.get("img_path", "output/volume_comparison.png")
        if Path(img_path).exists():
            self.img_comparison.set_image(img_path)

        meas = results.get("meas_real", {})
        for key in ["cuello", "pecho", "cintura", "cadera"]:
            if key in meas and key in self.meas_cards:
                self.meas_cards[key].setText(f"{meas[key]:.1f} cm")

        zone_stats = results.get("zone_stats", {})
        for zone, card in self.zone_cards.items():
            if zone in zone_stats:
                s = zone_stats[zone]
                card.update_values(s["mean_cm"], s["pct_excess"])

        # Guardar valores numéricos para PDF sin dependencia PyQt
        self._zone_values = {}
        for zone, card in self.zone_cards.items():
            try:
                txt = card.lbl_val.text()
                self._zone_values[zone] = float(
                    txt.replace("+", "").replace(" cm", "")
                )
            except ValueError:
                self._zone_values[zone] = 0.0

    def _on_pipeline_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Error en pipeline")
        QMessageBox.critical(self, "Error", msg[:800])

    # ── PDF ───────────────────────────────────────────────────────────────────
    def _export_pdf(self):
        if self._last_results is None:
            QMessageBox.warning(self, "Sin datos",
                                "Ejecuta el pipeline primero.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar reporte PDF",
            str(Path.home() / "Desktop" / "reporte_corporal.pdf"),
            "PDF (*.pdf)"
        )
        if not save_path:
            return

        try:
            from src.pdf_report import generate_pdf

            p = self._last_results.get("user_params", {})
            patient_data = {
                "name":     p.get("name",     "No especificado"),
                "sex":      "Masculino" if p.get("sex") == "male"
                            else "Femenino",
                "age":      p.get("age",      "—"),
                "weight":   p.get("weight",   "—"),
                "height":   p.get("height",   "—"),
                "body_fat": p.get("body_fat", "—"),
            }

            class SimpleCard:
                def __init__(self, val: float):
                    self._val = val
                    self.lbl_val = self

                def text(self) -> str:
                    sign = "+" if self._val >= 0 else ""
                    return f"{sign}{self._val:.2f} cm"

            simple_cards = {
                zone: SimpleCard(val)
                for zone, val in self._zone_values.items()
            }

            generate_pdf(
                img_path     = self._last_results.get(
                                   "img_path",
                                   "output/volume_comparison.png"),
                zone_cards   = simple_cards,
                output_path  = save_path,
                patient_data = patient_data,
                meas_real    = self._last_results.get("meas_real"),
                synth_meas   = self._last_results.get("synth_meas"),
            )
            self.lbl_status.setText("PDF exportado ✓")
            QMessageBox.information(
                self, "PDF exportado",
                f"Reporte guardado en:\n{save_path}"
            )
        except Exception as e:
            import traceback
            QMessageBox.critical(
                self, "Error al exportar PDF",
                traceback.format_exc()[:800]
            )