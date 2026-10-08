"""
main_window.py
--------------
GUI SomaRecon 3D — 3 páginas con navegación por tabs.

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
    QTabWidget, QDialog, QInputDialog, QCheckBox, QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

from src.gui.branding import (
    APP_NAME, app_icon, app_icon_pixmap, line_icon, icon_size
)

# ── Paleta ─────────────────────────────────────────────────────────────────────
# Identidad SomaRecon 3D: verde petróleo de marca, superficies frías y neutras,
# naranja para "mayor que la referencia" y azul para "menor que la referencia".
TEAL          = "#0B4F43"   # color de marca / acciones principales
TEAL_LIGHT    = "#D9ECE7"
TEAL_DARK     = "#083F35"
TEAL_MID      = "#17685A"   # pasos inactivos sobre la cabecera
MINT          = "#8FE3CD"   # acento claro sobre fondos oscuros
CORAL         = "#C2540F"   # exceso / detener / advertencia
CORAL_LIGHT   = "#FBEBDD"
PURPLE        = "#14232A"   # navegación hacia adelante (tinta)
PURPLE_LIGHT  = "#E6EBEE"
BLUE          = "#1F5F9E"   # déficit / acciones de IA (landmarks)
BLUE_LIGHT    = "#E4EEF8"
INK           = "#14232A"

BG_MAIN       = "#EEF2F3"   # fondo general de la ventana
BG_PANEL      = "#FFFFFF"   # tarjetas / paneles elevados
BG_CARD       = "#F4F7F8"   # inputs y superficies internas
BG_CARD_HOVER = "#E6EBEE"
BG_VIEWER     = "#0F1C22"   # visor oscuro de imágenes
BORDER        = "#DDE3E6"
BORDER_STRONG = "#C3CCD1"
TEXT_PRI      = "#14232A"
TEXT_SEC      = "#36464E"
TEXT_HINT     = "#6B7A83"
TEXT_ON_DARK  = "#FFFFFF"
TEXT_ON_VIEWER = "#C5D2D8"

SUCCESS = TEAL
DANGER  = CORAL
INFO    = BLUE

RADIUS    = 14
RADIUS_SM = 9
FONT_FAMILY = '"Figtree", "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
FONT_MONO   = '"IBM Plex Mono", "JetBrains Mono", "Consolas", "Menlo", monospace'

ZONES = ["cuello", "pecho", "brazo", "cintura", "cadera", "muslo", "rodilla"]

VIEW_LABELS = {
    "frontal": "Frontal", "posterior": "Posterior",
    "lateral_izq": "Lateral izq.", "lateral_der": "Lateral der.",
}
VIEW_TITLES = {
    "frontal": "Vista frontal", "posterior": "Vista posterior",
    "lateral_izq": "Vista lateral izquierda", "lateral_der": "Vista lateral derecha",
}

# Carpeta raíz del repositorio (main_window.py está en src/gui/, subimos 2 niveles)
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
CAPTURES_DIR   = PROJECT_ROOT / "data" / "capturas"


# ── Helpers de UI ──────────────────────────────────────────────────────────────
def _shade(hex_color, factor):
    """Aclara (factor>1) u oscurece (factor<1) un color hex conservando el tono."""
    c = QColor(hex_color)
    h, s, l, a = c.getHsl()
    l = max(0, min(255, int(round(l * factor))))
    c.setHsl(h, s, l, a)
    return c.name()


def hline():
    f = QFrame(); f.setFrameShape(QFrame.HLine); f.setFixedHeight(1)
    f.setStyleSheet(f"background:{BORDER};border:none;"); return f

def vline():
    f = QFrame(); f.setFrameShape(QFrame.VLine)
    f.setFixedWidth(1); f.setStyleSheet(f"background:{BORDER};border:none;"); return f

def section_label(text):
    l = QLabel(text.upper())
    l.setStyleSheet(
        f"font-size:11px;font-weight:600;color:{TEXT_HINT};letter-spacing:0.6px;"
        f"background:transparent;border:none;"
    )
    return l

def btn(text, fg=TEXT_PRI, bg=BG_PANEL, variant="soft", icon=None, icon_right=False):
    """Botón de la app.

    variant="soft"  → superficie clara con texto de color (acciones
                        secundarias). Sobre blanco lleva un borde fino.
    variant="solid" → botón relleno de color, para la acción principal de
                        cada pantalla (mayor peso visual).
    icon            → nombre de un icono de línea de branding.line_icon.
    """
    b = QPushButton(text)
    if variant == "solid":
        base   = bg if bg not in (BG_CARD, BG_PANEL) else fg
        hover  = _shade(base, 0.86)
        press  = _shade(base, 0.74)
        icon_color = TEXT_ON_DARK
        b.setStyleSheet(f"""
            QPushButton{{background:{base};color:{TEXT_ON_DARK};border:none;
                border-radius:{RADIUS_SM}px;padding:10px 14px;font-size:14px;font-weight:700;}}
            QPushButton:hover{{background:{hover};}}
            QPushButton:pressed{{background:{press};}}
            QPushButton:disabled{{background:{BORDER};color:{TEXT_HINT};}}
        """)
    else:
        hover = _shade(bg, 0.95)
        press = _shade(bg, 0.90)
        border = BORDER_STRONG if bg in (BG_CARD, BG_PANEL) else "transparent"
        icon_color = fg
        b.setStyleSheet(f"""
            QPushButton{{background:{bg};color:{fg};border:1px solid {border};
                border-radius:{RADIUS_SM}px;padding:9px 14px;font-size:14px;font-weight:600;}}
            QPushButton:hover{{background:{hover};}}
            QPushButton:pressed{{background:{press};}}
            QPushButton:disabled{{background:{BG_CARD};color:{TEXT_HINT};border-color:{BORDER};}}
        """)
    if icon:
        b.setIcon(line_icon(icon, icon_color))
        b.setIconSize(icon_size(18))
        if icon_right:
            b.setLayoutDirection(Qt.RightToLeft)
    b.setCursor(Qt.PointingHandCursor)
    b.setMinimumHeight(44)
    return b

def input_style():
    return (f"QWidget{{background:{BG_PANEL};border:1px solid {BORDER_STRONG};"
            f"border-radius:{RADIUS_SM}px;padding:8px 10px;font-size:14px;color:{TEXT_PRI};}}"
            f"QWidget:hover{{border-color:{TEXT_HINT};}}"
            f"QWidget:focus{{border-color:{TEAL};}}")

def labeled(label_text, widget):
    c = QWidget(); l = QVBoxLayout(c)
    l.setContentsMargins(0,0,0,0); l.setSpacing(5)
    lb = QLabel(label_text)
    lb.setStyleSheet(f"font-size:13px;color:{TEXT_SEC};font-weight:500;"
                     f"background:transparent;border:none;")
    l.addWidget(lb); l.addWidget(widget); return c

def card_panel(radius=RADIUS, bg=BG_PANEL, border=True):
    """Estilo de tarjeta elevada reutilizable para los paneles principales."""
    b = f"border:1px solid {BORDER};" if border else "border:none;"
    return f"background:{bg};border-radius:{radius}px;{b}"

def elevate(widget, blur=24, y_offset=4, alpha=28):
    """Aplica una sombra suave a un panel/tarjeta para dar sensación de profundidad."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setXOffset(0)
    shadow.setYOffset(y_offset)
    shadow.setColor(QColor(20, 35, 42, alpha))
    widget.setGraphicsEffect(shadow)
    return widget

_card_seq = [0]

def card(margins=(20, 20, 20, 20), spacing=12):
    """Tarjeta blanca con sombra. Devuelve (QFrame, QVBoxLayout).

    Usa un objectName único para que el fondo/borde NO se herede a los
    widgets hijos (un stylesheet sin selector se aplicaría a todos ellos).
    """
    _card_seq[0] += 1
    f = QFrame()
    f.setObjectName(f"card{_card_seq[0]}")
    f.setStyleSheet(
        f"#{f.objectName()}{{background:{BG_PANEL};border:none;border-radius:{RADIUS}px;}}"
    )
    elevate(f, blur=18, y_offset=2, alpha=22)
    lay = QVBoxLayout(f)
    lay.setContentsMargins(*margins); lay.setSpacing(spacing)
    return f, lay

def text_label(text, size=14, color=TEXT_PRI, weight=400, mono=False, wrap=False):
    l = QLabel(text)
    fam = f"font-family:{FONT_MONO};" if mono else ""
    l.setStyleSheet(f"{fam}font-size:{size}px;font-weight:{weight};color:{color};"
                    f"background:transparent;border:none;")
    l.setWordWrap(wrap)
    return l

def card_title(text, icon=None):
    """Título de tarjeta con icono de línea opcional. Devuelve un QWidget."""
    w = QWidget(); w.setStyleSheet("background:transparent;")
    h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0); h.setSpacing(8)
    if icon:
        ic = QLabel(); ic.setPixmap(line_icon(icon, TEAL).pixmap(icon_size(20)))
        ic.setStyleSheet("background:transparent;border:none;")
        h.addWidget(ic)
    h.addWidget(text_label(text, 16, TEXT_PRI, 700))
    h.addStretch()
    return w

def thin_bar(color=TEAL):
    """Barra de progreso fina (8 px) sin texto."""
    bar = QProgressBar(); bar.setRange(0, 100); bar.setValue(0)
    bar.setTextVisible(False); bar.setFixedHeight(8)
    set_bar_color(bar, color)
    return bar

def set_bar_color(bar, color):
    bar.setStyleSheet(f"""
        QProgressBar{{background:#E6EBEE;border-radius:4px;border:none;}}
        QProgressBar::chunk{{background:{color};border-radius:4px;}}
    """)


class Segmented(QFrame):
    """Control segmentado (grupo de botones excluyentes), p. ej. RGB | Profundidad."""

    changed = pyqtSignal(int)

    def __init__(self, labels=()):
        super().__init__()
        self.setObjectName("segmented")
        self.setStyleSheet(
            "#segmented{background:#DBE2E5;border-radius:10px;border:none;}"
        )
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(4, 4, 4, 4); self._lay.setSpacing(4)
        self._group = QButtonGroup(self); self._group.setExclusive(True)
        self._buttons = []
        self.set_options(labels)

    def set_options(self, labels):
        for b in self._buttons:
            self._group.removeButton(b); b.setParent(None); b.deleteLater()
        self._buttons = []
        for i, text in enumerate(labels):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(38)
            b.setStyleSheet(f"""
                QPushButton{{background:transparent;color:{TEXT_SEC};border:none;
                    border-radius:8px;padding:6px 16px;font-size:14px;font-weight:600;}}
                QPushButton:hover:!checked{{background:#CFD8DC;}}
                QPushButton:checked{{background:{BG_PANEL};color:{TEXT_PRI};}}
            """)
            b.clicked.connect(lambda _, idx=i: self.changed.emit(idx))
            self._group.addButton(b, i)
            self._lay.addWidget(b)
            self._buttons.append(b)
        if self._buttons:
            self._buttons[0].setChecked(True)

    def set_current(self, idx):
        if 0 <= idx < len(self._buttons):
            self._buttons[idx].setChecked(True)


def _bgr_to_pixmap(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, c = rgb.shape
    qi = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qi.copy())


class _ViewSlot:
    """Guarda la última imagen de una vista (RGB o profundidad).

    Sustituye a los antiguos widgets por vista: el resto del código sigue
    llamando a `.set_image_array(img)`, y la página decide si esa imagen se
    muestra en el visor grande (según la vista y el modo activos).
    """

    def __init__(self, on_change):
        self.array = None
        self._on_change = on_change

    def set_image_array(self, img_bgr):
        self.array = img_bgr
        self._on_change()


class ViewThumb(QFrame):
    """Miniatura de una vista en la pantalla de Captura."""

    clicked = pyqtSignal(str)

    def __init__(self, name):
        super().__init__()
        self._name = name
        self.setObjectName("thumb")
        self.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)
        self.img = QLabel("Pendiente")
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setFixedHeight(84)
        lay.addWidget(self.img)
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(text_label(VIEW_LABELS.get(name, name), 14, TEXT_PRI, 700))
        row.addStretch()
        self.check = QLabel()
        self.check.setFixedSize(22, 22)
        self.check.setAlignment(Qt.AlignCenter)
        self.check.setPixmap(line_icon("check", "#FFFFFF", 14, 3.5).pixmap(icon_size(14)))
        self.check.setStyleSheet(f"background:{TEAL};border-radius:11px;border:none;")
        self.check.setVisible(False)
        row.addWidget(self.check)
        lay.addLayout(row)
        self._has_image = False
        self.set_state(active=False, done=False)

    def set_image(self, img_bgr):
        pm = _bgr_to_pixmap(img_bgr).scaledToHeight(84, Qt.SmoothTransformation)
        self.img.setPixmap(pm)
        self._has_image = True

    def clear_image(self):
        self.img.setPixmap(QPixmap())
        self._has_image = False

    def set_state(self, active, done):
        self.check.setVisible(done)
        if active:
            border = f"2px solid {TEAL}"
        elif done:
            border = f"1px solid {BORDER_STRONG}"
        else:
            border = f"1px dashed {TEXT_HINT}"
        self.setStyleSheet(
            f"#thumb{{background:{BG_PANEL};border:{border};border-radius:12px;}}"
        )
        if self._has_image:
            self.img.setStyleSheet("background:#22343D;border-radius:8px;border:none;")
        else:
            self.img.setText("Capturando" if active else "Pendiente")
            bg, fg = (TEAL_LIGHT, TEAL) if active else (BG_MAIN, TEXT_SEC)
            self.img.setStyleSheet(
                f"background:{bg};color:{fg};border-radius:8px;border:none;"
                f"font-size:13px;font-weight:600;"
            )

    def mousePressEvent(self, e):
        self.clicked.emit(self._name)
        super().mousePressEvent(e)


# ── Estado compartido ──────────────────────────────────────────────────────────
class AppState:
    """Estado global compartido entre las 3 páginas."""
    def __init__(self):
        self.n_views    = 1
        self.view_names = ["frontal"]
        self.rgbs       = {}
        self.depths     = {}
        self.roi_params = {}
        self.only_neck_mode = False  # True → mediciones y comparación usan
                                      # exclusivamente la circunferencia del
                                      # cuello (ver Page2Measurements, checkbox
                                      # "Solo cuello", y Page3Results._run_smpl)
        self.patient    = {
            "name": "", "sex": "female", "age": 30,
            "weight": 65.0, "height": 1.65, "body_fat": 25.0,
        }
        self.segs         = {}
        # NOTA: STATE ya no guarda pcds/meshes/fused_pcd/fused_mesh — el
        # pipeline (Ejecutar pipeline) solo segmenta y mide; la
        # reconstrucción de malla ahora se genera bajo demanda al
        # presionar "Comparar con Malla de Referencia" en Resultados
        # (ver Page3Results._run_smpl), no se guarda en STATE.
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
    """QLabel que acepta líneas horizontales de medición superpuestas.

    dark=True → visor oscuro con degradado (pantallas de Captura y Mediciones).
    """

    def __init__(self, dark=False):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        if dark:
            self.setStyleSheet(
                "background:qradialgradient(cx:0.5,cy:0.38,radius:0.85,fx:0.5,fy:0.38,"
                f"stop:0 #2C4350,stop:1 {BG_VIEWER});border-radius:16px;"
                f"color:{TEXT_ON_VIEWER};font-size:15px;font-weight:500;"
            )
        else:
            self.setStyleSheet(f"background:{BG_CARD};border-radius:6px;")
        self.setMinimumSize(200, 200)
        self._lines   = []
        self._pixmap_base = None

    def set_image_array(self, img_bgr: np.ndarray):
        self._pixmap_base = _bgr_to_pixmap(img_bgr)
        self._redraw()

    def clear_image(self, text=""):
        """Quita la imagen y muestra un texto de ayuda en su lugar."""
        self._pixmap_base = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def set_lines(self, lines):
        """lines: list of (y_norm 0-1, color_hex, label_str)"""
        self._lines = lines
        self._redraw()

    def _redraw(self):
        if self._pixmap_base is None:
            return
        pad = 24
        pix = self._pixmap_base.scaled(
            max(10, self.width() - pad), max(10, self.height() - pad),
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
            from src.reconstruction      import FY

            segs = {}

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
                segs[name] = seg

            STATE.segs = segs

            # NOTA: el pipeline de "Ejecutar pipeline" (aquí) solo ubica las
            # zonas de interés y calcula las medidas antropométricas — ya
            # NO reconstruye mallas 3D ni las fusiona entre vistas. Eso es
            # deliberado: la reconstrucción de malla ahora ocurre solo al
            # presionar "Comparar con Malla de Referencia" en la página de
            # Resultados (ver Page3Results._run_smpl), y usa un método
            # distinto (medidas → malla vía reshaper, no fusión ICP de
            # nubes de puntos), que es más rápido, más confiable y no
            # depende de que los ángulos de captura de las 4 vistas estén
            # bien calibrados. La fusión ICP + Poisson que vivía aquí antes
            # además podía crashear el hilo (error nativo de Open3D/
            # PoissonRecon 'Failed to close loop' con fusiones de baja
            # calidad) — quitarla de este paso también resuelve eso.

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
                # Pasar RGB de todas las vistas para deteccion automatica
                meas = combine_measurements(
                    _mk("frontal"), _mk("posterior"),
                    _mk("lateral_izq"), _mk("lateral_der"),
                    rgb_image       = STATE.rgbs.get("frontal"),
                    rgb_posterior   = STATE.rgbs.get("posterior"),
                    rgb_lateral_izq = STATE.rgbs.get("lateral_izq"),
                    rgb_lateral_der = STATE.rgbs.get("lateral_der"),
                )

                # STATE.measurements guarda el dict completo (no solo circ_cm)
                # Estructura: {zona: {"y_px", "circumference_cm", "width_mm", ...}}
                # Si "Solo cuello" está activo, el pipeline no debe medir (ni
                # mostrar) el resto de zonas — antes se calculaban todas
                # igual y "Solo cuello" solo afectaba el paso de comparación
                # con la malla (Página 3), lo cual era confuso: la página
                # de Mediciones seguía mostrando pecho/cintura/etc.
                if STATE.only_neck_mode:
                    meas = {"cuello": meas.get("cuello")}
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
        self._active_view = "frontal"
        self._mode        = "rgb"
        self._thumbs      = {}
        self._roi_shape   = None

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(20)

        # ── Columna izquierda: datos del paciente ───────────────────────────
        left, ll = card()
        left.setFixedWidth(300)
        ll.addWidget(card_title("Datos del paciente", "user"))

        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText("Nombre del paciente")
        self.inp_name.setStyleSheet(input_style())
        ll.addWidget(labeled("Nombre", self.inp_name))

        r1 = QHBoxLayout(); r1.setSpacing(12)
        self.cmb_sex = QComboBox()
        self.cmb_sex.addItems(["Femenino","Masculino"])
        self.inp_age = QLineEdit("30"); self.inp_age.setStyleSheet(input_style())
        r1.addWidget(labeled("Sexo", self.cmb_sex))
        r1.addWidget(labeled("Edad", self.inp_age))
        ll.addLayout(r1)

        r2 = QHBoxLayout(); r2.setSpacing(12)
        self.inp_weight = QLineEdit("65.0"); self.inp_weight.setStyleSheet(input_style())
        self.inp_height = QLineEdit("1.65"); self.inp_height.setStyleSheet(input_style())
        r2.addWidget(labeled("Peso (kg)", self.inp_weight))
        r2.addWidget(labeled("Talla (m)", self.inp_height))
        ll.addLayout(r2)

        bf_head = QHBoxLayout()
        bf_head.addWidget(text_label("% grasa corporal", 13, TEXT_SEC, 500))
        bf_head.addStretch()
        self.lbl_bf = text_label("25 %", 14, TEXT_PRI, 600, mono=True)
        bf_head.addWidget(self.lbl_bf)
        ll.addLayout(bf_head)
        self.slider_bf = QSlider(Qt.Horizontal); self.slider_bf.setRange(5,45); self.slider_bf.setValue(25)
        self.slider_bf.setMinimumHeight(28)
        self.slider_bf.valueChanged.connect(lambda v: self.lbl_bf.setText(f"{v} %"))
        ll.addWidget(self.slider_bf)

        self.cmb_views = QComboBox()
        self.cmb_views.addItems(["1 vista (frontal)",
                                  "2 vistas (frontal + posterior)",
                                  "3 vistas (+ lateral izq.)",
                                  "4 vistas (completo)"])
        self.cmb_views.currentIndexChanged.connect(self._on_n_views_changed)
        ll.addWidget(labeled("Número de vistas", self.cmb_views))
        ll.addStretch()
        root.addWidget(left)

        # ── Centro: visor grande + miniaturas ───────────────────────────────
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(14)

        head = QHBoxLayout()
        self.lbl_view_title = text_label("Vista frontal", 24, TEXT_PRI, 700)
        head.addWidget(self.lbl_view_title)
        head.addStretch()
        self.seg_mode = Segmented(["RGB", "Profundidad"])
        self.seg_mode.changed.connect(self._on_mode_changed)
        head.addWidget(self.seg_mode)
        cv.addLayout(head)

        self.viewer = ImageWithOverlay(dark=True)
        self.viewer.setMinimumHeight(260)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cv.addWidget(self.viewer, stretch=1)

        self.thumb_row = QHBoxLayout()
        self.thumb_row.setSpacing(12)
        cv.addLayout(self.thumb_row)
        root.addWidget(center, stretch=1)

        # ── Columna derecha: acciones de captura ────────────────────────────
        right = QWidget(); right.setFixedWidth(300)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(14)

        cap, cl = card()
        cap_head = QHBoxLayout()
        cap_head.addWidget(text_label("Captura", 16, TEXT_PRI, 700))
        cap_head.addStretch()
        self.lbl_progress = text_label("0 de 1 vistas", 13, TEXT_SEC, 500)
        cap_head.addWidget(self.lbl_progress)
        cl.addLayout(cap_head)
        self.bar_progress = thin_bar(TEAL)
        cl.addWidget(self.bar_progress)

        self.btn_capture = btn("Capturar vista", TEAL, TEAL, variant="solid", icon="camera")
        self.btn_capture.setMinimumHeight(52)
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self._capture_frame)
        cl.addWidget(self.btn_capture)

        self.btn_import = btn("Importar imágenes", icon="upload")
        self.btn_import.clicked.connect(self._import_images)
        cl.addWidget(self.btn_import)

        self.btn_cam_start = btn("Iniciar cámara", icon="play")
        self.btn_cam_stop  = btn("Detener cámara", CORAL, BG_PANEL, icon="stop")
        self.btn_cam_stop.setEnabled(False)
        self.btn_cam_stop.setVisible(False)
        self.btn_cam_start.clicked.connect(self._start_camera)
        self.btn_cam_stop.clicked.connect(self._stop_camera)
        cl.addWidget(self.btn_cam_start)
        cl.addWidget(self.btn_cam_stop)
        rv.addWidget(cap)

        # Ajustes de región (ROI) y profundidad de la vista activa, plegables
        roi_card, rl = card(margins=(20, 12, 20, 12), spacing=8)
        self.btn_roi_toggle = QPushButton("Ajustes de región y profundidad")
        self.btn_roi_toggle.setCheckable(True)
        self.btn_roi_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_roi_toggle.setIcon(line_icon("sliders", TEXT_PRI))
        self.btn_roi_toggle.setIconSize(icon_size(18))
        self.btn_roi_toggle.setMinimumHeight(40)
        self.btn_roi_toggle.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;text-align:left;"
            f"font-size:14px;font-weight:600;color:{TEXT_PRI};padding:4px 0;}}"
            f"QPushButton:hover{{color:{TEAL};}}"
        )
        rl.addWidget(self.btn_roi_toggle)
        self.roi_box = QWidget()
        self.roi_box.setStyleSheet("background:transparent;")
        self.roi_grid = QGridLayout(self.roi_box)
        self.roi_grid.setContentsMargins(0, 0, 0, 4)
        self.roi_grid.setHorizontalSpacing(8); self.roi_grid.setVerticalSpacing(6)
        self.roi_box.setVisible(False)
        self.btn_roi_toggle.toggled.connect(self.roi_box.setVisible)
        rl.addWidget(self.roi_box)
        rv.addWidget(roi_card)

        self.lbl_status = text_label("En espera", 13, TEXT_SEC, 400, wrap=True)
        rv.addWidget(self.lbl_status)
        rv.addStretch()

        self.btn_next = btn("Continuar a Mediciones", INK, INK, variant="solid",
                            icon="arrow-right", icon_right=True)
        self.btn_next.setMinimumHeight(52)
        self.btn_next.clicked.connect(self._save_patient_and_next)
        rv.addWidget(self.btn_next)
        root.addWidget(right)

        # Selector interno de "vista actual de la cámara". Ya no se muestra:
        # la vista activa se elige con las miniaturas, y este combo solo se
        # mantiene sincronizado para el código de cámara que lo consulta.
        self.cmb_cam_view = QComboBox(self)
        self.cmb_cam_view.setVisible(False)
        self.cmb_cam_view.addItems(STATE.view_names)
        self.cmb_cam_view.currentTextChanged.connect(self._on_cam_view_changed)

        self._on_n_views_changed(0)

        self.cam = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def _on_n_views_changed(self, idx):
        STATE.set_n_views(idx + 1)
        # Refrescar el combo interno de "vista actual de la cámara"
        if hasattr(self, "cmb_cam_view"):
            self.cmb_cam_view.blockSignals(True)
            self.cmb_cam_view.clear()
            self.cmb_cam_view.addItems(STATE.view_names)
            self.cmb_cam_view.blockSignals(False)
        self._rebuild_view_widgets()

    def _on_cam_view_changed(self, name):
        if name and name != self._active_view:
            self._set_active_view(name)

    def _on_mode_changed(self, idx):
        self._mode = "rgb" if idx == 0 else "depth"
        self._refresh_viewer()

    def _rebuild_view_widgets(self):
        """Reconstruye las miniaturas y los almacenes de imagen por vista."""
        while self.thumb_row.count():
            item = self.thumb_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        self._view_widgets = {}
        self._thumbs       = {}

        for name in STATE.view_names:
            self._view_widgets[name] = {
                "rgb":   _ViewSlot(lambda n=name: self._on_slot_changed(n, "rgb")),
                "depth": _ViewSlot(lambda n=name: self._on_slot_changed(n, "depth")),
            }
            thumb = ViewThumb(name)
            thumb.clicked.connect(self._set_active_view)
            self.thumb_row.addWidget(thumb, stretch=1)
            self._thumbs[name] = thumb

        if self._active_view not in STATE.view_names:
            self._active_view = STATE.view_names[0]

        # Recuperar las imágenes que ya estaban cargadas para estas vistas
        for name in STATE.view_names:
            rgb = STATE.rgbs.get(name)
            if rgb is not None:
                self._view_widgets[name]["rgb"].set_image_array(rgb)
            self._update_overlay(name)

        self._set_active_view(self._active_view)

    def _set_active_view(self, name):
        """Cambia la vista mostrada en el visor (y a la que apunta la cámara)."""
        if name not in STATE.view_names:
            return
        self._active_view      = name
        self._current_cam_view = name
        self.lbl_view_title.setText(VIEW_TITLES.get(name, name))
        self.cmb_cam_view.blockSignals(True)
        self.cmb_cam_view.setCurrentText(name)
        self.cmb_cam_view.blockSignals(False)
        self._build_roi_panel(name)
        self._refresh_thumbs()
        self._refresh_viewer()

    def _on_slot_changed(self, name, kind):
        """Llega cada vez que cambia la imagen RGB o de profundidad de una vista."""
        slot = self._view_widgets.get(name, {}).get(kind)
        if slot is None:
            return
        if kind == "rgb" and name in self._thumbs and slot.array is not None:
            self._thumbs[name].set_image(slot.array)
        if self._ensure_roi_scaled(name) and name == self._active_view:
            self._build_roi_panel(name)
        self._refresh_thumbs()
        if name == self._active_view and kind == self._mode:
            self._refresh_viewer()

    def _refresh_thumbs(self):
        done_count = 0
        for name, thumb in self._thumbs.items():
            done = STATE.rgbs.get(name) is not None
            done_count += int(done)
            thumb.set_state(active=(name == self._active_view), done=done)
        total = max(1, len(self._thumbs))
        plural = "vista" if total == 1 else "vistas"
        self.lbl_progress.setText(f"{done_count} de {total} {plural}")
        self.bar_progress.setValue(int(round(done_count / total * 100)))

    def _refresh_viewer(self):
        slot = self._view_widgets.get(self._active_view, {}).get(self._mode)
        if slot is not None and slot.array is not None:
            self.viewer.set_image_array(slot.array)
        else:
            kind = "RGB" if self._mode == "rgb" else "de profundidad"
            self.viewer.clear_image(
                f"Sin imagen {kind} para esta vista.\n"
                "Inicia la cámara o importa imágenes."
            )

    def _image_size(self, name):
        """(ancho, alto) de la imagen real de esta vista, o None si no hay."""
        src = STATE.rgbs.get(name)
        if src is None:
            src = STATE.depths.get(name)
        if src is None:
            slot = self._view_widgets.get(name, {}).get("rgb")
            src = slot.array if slot is not None else None
        if src is None:
            return None
        h, w = src.shape[:2]
        return (w, h)

    def _ensure_roi_scaled(self, name):
        """
        Los valores por defecto de roi_params (x1/x2/y1/y2) fueron calibrados
        sobre una imagen de 1280x720. Cuando se conoce el tamaño real de la
        imagen de esta vista, se reescalan una sola vez para que la región
        inicial quede en una posición razonable. Devuelve True si reescaló.
        """
        size = self._image_size(name)
        roi  = STATE.roi_params.get(name)
        if size is None or roi is None:
            return False
        prev = roi.get("_scaled_wh")
        if prev == size:
            return False
        ref_w, ref_h = prev if prev else (1280, 720)
        sx, sy = size[0] / ref_w, size[1] / ref_h
        roi["x1"] = int(roi["x1"] * sx)
        roi["x2"] = int(roi["x2"] * sx)
        roi["y1"] = int(roi["y1"] * sy)
        roi["y2"] = int(roi["y2"] * sy)
        roi["_scaled_wh"] = size
        return True

    def _build_roi_panel(self, name):
        """Deslizadores de región (x/y) y profundidad para la vista activa."""
        while self.roi_grid.count():
            item = self.roi_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._roi_sliders = {}

        self._ensure_roi_scaled(name)
        roi  = STATE.roi_params[name]
        size = self._image_size(name)
        # El % de cada deslizador se traduce a píxeles con el tamaño REAL de
        # la imagen cargada (si aún no hay imagen, se asume 1280x720, que es
        # la resolución a la que están calibrados los valores por defecto).
        w_img, h_img = size if size else (1280, 720)

        params_def = [
            ("X mín %",       "x1",    int(roi["x1"]/w_img*100), 0,   100),
            ("X máx %",       "x2",    int(roi["x2"]/w_img*100), 0,   100),
            ("Y mín %",       "y1",    int(roi["y1"]/h_img*100), 0,   100),
            ("Y máx %",       "y2",    int(roi["y2"]/h_img*100), 0,   100),
            ("Prof. mín cm",  "d_min", roi["d_min"]//10,          50, 250),
            ("Prof. máx cm",  "d_max", roi["d_max"]//10,          50, 400),
        ]
        sliders = {}
        for row_i, (label_txt, key, val, lo, hi) in enumerate(params_def):
            lbl = text_label(label_txt, 13, TEXT_SEC, 500)
            sl  = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi); sl.setValue(val)
            sl.setMinimumHeight(26)
            val_lbl = text_label(str(val), 13, TEXT_PRI, 600, mono=True)
            val_lbl.setFixedWidth(34)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            def _make_cb(k, vl, nm, w_i=w_img, h_i=h_img):
                def cb(v):
                    vl.setText(str(v))
                    if k in ("x1","x2"):
                        STATE.roi_params[nm][k] = int(v/100*w_i)
                    elif k in ("y1","y2"):
                        STATE.roi_params[nm][k] = int(v/100*h_i)
                    else:
                        STATE.roi_params[nm][k] = v * 10
                    self._update_overlay(nm)
                return cb

            sl.valueChanged.connect(_make_cb(key, val_lbl, name))
            sliders[key] = sl
            self.roi_grid.addWidget(lbl,     row_i, 0)
            self.roi_grid.addWidget(sl,      row_i, 1)
            self.roi_grid.addWidget(val_lbl, row_i, 2)
        self.roi_grid.setColumnStretch(1, 1)
        self._roi_sliders[name] = sliders

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
        self.btn_cam_start.setVisible(False)
        self.btn_cam_stop.setVisible(True)
        self.btn_capture.setEnabled(True)
        self.lbl_status.setText(f"Cámara activa — {mode}")
        # Respeta la vista que el usuario ya tenía seleccionada en el combo
        self._current_cam_view = self.cmb_cam_view.currentText() or STATE.view_names[0]

    def _stop_camera(self):
        self.timer.stop()
        if self.cam: self.cam.stop(); self.cam = None
        self.btn_cam_start.setEnabled(True)
        self.btn_cam_stop.setEnabled(False)
        self.btn_cam_start.setVisible(True)
        self.btn_cam_stop.setVisible(False)
        self.btn_capture.setEnabled(False)
        self.lbl_status.setText("Cámara detenida")

    def _update_frame(self):
        if not self.cam: return
        try:
            rgb, depth = self.cam.get_frames()
            # Colormap de profundidad para visualización en vivo (antes esto
            # nunca se calculaba ni se mostraba, por eso solo se veía el RGB)
            depth_vis = self.cam.get_depth_colormap(depth)
            name = getattr(self, "_current_cam_view", STATE.view_names[0])
            if name in self._view_widgets:
                self._view_widgets[name]["rgb"].set_image_array(rgb)
                self._view_widgets[name]["depth"].set_image_array(depth_vis)
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

        # ── Pedir nombre y guardar en disco ─────────────────────────────────
        # Se guardan ambas imágenes (RGB + Depth) dentro de una carpeta,
        # en data/capturas/<nombre>/, dentro del propio repositorio.
        folder_name, ok = QInputDialog.getText(
            self,
            "Guardar captura",
            f"Nombre de la carpeta para guardar la captura de '{name}'\n"
            f"(se guardará en: {CAPTURES_DIR}):",
            text=name
        )
        if ok and folder_name.strip():
            try:
                saved_dir = self._save_capture_to_disk(
                    name, rgb, depth, folder_name.strip()
                )
                self.lbl_status.setText(
                    f"Frame {name} capturado ✓  —  guardado en {saved_dir}"
                )
            except Exception as e:
                self.lbl_status.setText(f"Frame {name} capturado, pero falló el guardado: {e}")
        else:
            self.lbl_status.setText(f"Frame {name} capturado ✓ (no se guardó en disco)")

    def _save_capture_to_disk(self, view_name, rgb, depth, folder_name):
        """
        Guarda el par RGB + Depth de una captura dentro de:
            <repositorio>/data/capturas/<folder_name>/

        Archivos generados:
            <view_name>_rgb.png        → imagen a color
            <view_name>_depth.npy      → profundidad cruda en metros (float32)
            <view_name>_depth_vis.png  → colormap de profundidad (solo para revisar visualmente)

        Devuelve la ruta de la carpeta donde se guardó.
        """
        out_dir = CAPTURES_DIR / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        rgb_path   = out_dir / f"{view_name}_rgb.png"
        depth_path = out_dir / f"{view_name}_depth.npy"
        cv2.imwrite(str(rgb_path), rgb)
        np.save(str(depth_path), depth)

        if self.cam is not None:
            depth_vis = self.cam.get_depth_colormap(depth)
            cv2.imwrite(str(out_dir / f"{view_name}_depth_vis.png"), depth_vis)

        return out_dir

    def _reset_pipeline_state(self):
        """Limpia el estado derivado para que el pipeline se pueda re-ejecutar."""
        STATE.rgbs.clear()
        STATE.depths.clear()
        STATE.segs.clear()
        STATE.measurements.clear()
        STATE.diag.clear()
        STATE.pipeline_done = False
        STATE.height_cm     = 0.0
        STATE.smpl_result   = None
        try:
            from src.multi_view_pipeline import VIEW_ZONES
            for v in VIEW_ZONES:
                VIEW_ZONES[v].clear()
        except Exception:
            pass

    def _import_images(self):
        data_dir = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta con imágenes",
            str(Path.home())
        )
        if not data_dir:
            return

        # FIX: limpiar estado previo antes de cargar nuevas imágenes
        self._reset_pipeline_state()

        data_dir = Path(data_dir)
        loaded = []
        for name in STATE.view_names:
            rgb_path   = data_dir / f"{name}_rgb.png"
            depth_path = data_dir / f"{name}_depth.npy"
            if rgb_path.exists() and depth_path.exists():
                from src.loader import ensure_millimetres
                STATE.rgbs[name]   = cv2.imread(str(rgb_path))
                depth_raw          = np.load(str(depth_path)).astype("float32")
                STATE.depths[name] = ensure_millimetres(depth_raw, source=str(depth_path))
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
                from src.loader import ensure_millimetres
                depth_raw           = np.load(depth_path).astype("float32")
                STATE.depths[name]  = ensure_millimetres(depth_raw, source=depth_path)
            self._view_widgets[name]["rgb"].set_image_array(STATE.rgbs[name])
            self._update_overlay(name)
        self.lbl_status.setText("Importación completada ✓")

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

        # Dimensiones de la imagen COMPLETA (no del recorte) — las usan los
        # sliders ROI del panel derecho, que trabajan en % sobre la imagen
        # completa para poder ajustar/expandir la zona delimitada.
        _depth_full = STATE.depths.get(view_name)
        if _depth_full is not None:
            self._real_h, self._real_w = _depth_full.shape[:2]
        else:
            self._real_h, self._real_w = self._orig_rgb.shape[:2]

        # Recortar a la sección ROI (x1/x2/y1/y2) fijada en Captura — antes
        # esta ventana siempre mostraba el depth COMPLETO como base, sin
        # importar la zona delimitada manualmente.
        self._crop_x1, self._crop_y1, self._crop_x2, self._crop_y2 = \
            self._compute_crop_bounds()

        # Usar depth (recortado) como imagen base del canvas (en vez del RGB)
        depth_src = STATE.depths.get(view_name)
        if depth_src is not None:
            depth_crop = depth_src[self._crop_y1:self._crop_y2,
                                    self._crop_x1:self._crop_x2]
            valid = depth_crop[(depth_crop > 300) & (depth_crop < 4000)]
            d_norm = np.zeros_like(depth_crop, dtype=np.uint8)
            if len(valid) > 0:
                mask_v = (depth_crop > 300) & (depth_crop < 4000)
                d_norm[mask_v] = ((depth_crop[mask_v]-valid.min())/(valid.max()-valid.min())*255).astype(np.uint8)
            base_img = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
            # Guardar como RGB para consistencia con _orig_rgb
            self._orig_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        else:
            self._orig_rgb = image_rgb.copy()

        h, w = self._orig_rgb.shape[:2]   # dimensiones del RECORTE mostrado
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

    def _compute_crop_bounds(self):
        """
        (x1,y1,x2,y2) del ROI de self.view_name en coordenadas ABSOLUTAS de
        la imagen completa, ordenados y recortados a límites válidos.
        Si no hay depth cargado o el ROI es degenerado, devuelve el
        rectángulo completo (0,0,W,H).
        """
        depth = STATE.depths.get(self.view_name)
        if depth is None:
            return (0, 0, self._orig_rgb.shape[1], self._orig_rgb.shape[0])
        h, w = depth.shape[:2]
        roi = STATE.roi_params.get(self.view_name)
        if not roi:
            return (0, 0, w, h)
        x1, x2 = sorted((roi["x1"], roi["x2"]))
        y1, y2 = sorted((roi["y1"], roi["y2"]))
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return (0, 0, w, h)
        return (x1, y1, x2, y2)

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
            f"font-size:15px;font-weight:500;padding:6px;"
            f"border-radius:6px;background:{BG_CARD};color:{TEXT_PRI};"
        )
        ll.addWidget(self.lbl_instr)

        # Solo canvas depth — interactivo para marcar zonas
        dep_col = QVBoxLayout(); dep_col.setSpacing(2)
        dep_title = QLabel("Depth")
        dep_title.setStyleSheet(f"font-size:12px;color:{TEXT_HINT};")
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
        right.setStyleSheet(card_panel())

        elevate(right, blur=20, y_offset=2, alpha=10)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12,12,12,12); rl.setSpacing(8)

        rl.addWidget(section_label(f"Vista: {self.view_name}"))
        rl.addWidget(hline())

        rl.addWidget(section_label("Ajuste ROI y profundidad"))
        roi = STATE.roi_params[self.view_name]
        self._roi_sliders = {}
        # Dimensiones REALES de la imagen (no un valor fijo asumido), para que
        # el % del slider siempre corresponda al píxel correcto sin importar
        # la resolución con la que se capturó/importó la imagen.
        h_img, w_img = self._real_h, self._real_w

        params = [
            ("x1 %",  "x1",   int(roi["x1"]/w_img*100), 0,   100),
            ("x2 %",  "x2",   int(roi["x2"]/w_img*100), 0,   100),
            ("y1 %",  "y1",   int(roi["y1"]/h_img*100), 0,   100),
            ("y2 %",  "y2",   int(roi["y2"]/h_img*100), 0,   100),
            ("d_min", "d_min",roi["d_min"]//10,          50, 250),
            ("d_max", "d_max",roi["d_max"]//10,          50, 400),
        ]
        for label_txt, key, val, lo, hi in params:
            row = QHBoxLayout()
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(f"font-size:12px;color:{TEXT_SEC};min-width:48px;")
            sl  = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi); sl.setValue(val)
            vl  = QLabel(str(val))
            vl.setFixedWidth(28)
            vl.setStyleSheet(f"font-size:12px;color:{TEXT_PRI};")
            def _cb(v, k=key, vlbl=vl, w_i=w_img, h_i=h_img):
                vlbl.setText(str(v))
                if k in ("x1","x2"):
                    STATE.roi_params[self.view_name][k] = int(v/100*w_i)
                elif k in ("y1","y2"):
                    STATE.roi_params[self.view_name][k] = int(v/100*h_i)
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
            lbl.setStyleSheet(f"font-size:13px;color:{TEXT_SEC};")
            status = QLabel("✓" if z in self.zone_rects else "○")
            status.setStyleSheet(
                f"font-size:13px;color:{TEAL if z in self.zone_rects else TEXT_HINT};font-weight:700;"
            )
            row.addWidget(dot); row.addWidget(lbl); row.addStretch()
            row.addWidget(status)
            rl.addLayout(row)
            self._zone_list_labels[z] = status

        rl.addWidget(hline())
        rl.addStretch()

        btn_landmarks = btn("Detectar landmarks (MediaPipe)", BLUE, BLUE_LIGHT)
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
        """Regenera el canvas depth (recortado a la ROI actual) con los
        parámetros ROI actuales — se llama cuando el usuario mueve los
        sliders ROI de este diálogo."""
        depth = STATE.depths.get(self.view_name)
        if depth is None:
            return
        # Recalcular el recorte por si el usuario movió los sliders ROI
        self._crop_x1, self._crop_y1, self._crop_x2, self._crop_y2 = \
            self._compute_crop_bounds()
        depth_crop = depth[self._crop_y1:self._crop_y2,
                            self._crop_x1:self._crop_x2]
        valid = depth_crop[(depth_crop > 300) & (depth_crop < 4000)]
        d_norm = np.zeros_like(depth_crop, dtype=np.uint8)
        if len(valid) > 0:
            mask_v = (depth_crop > 300) & (depth_crop < 4000)
            d_norm[mask_v] = ((depth_crop[mask_v]-valid.min())/(valid.max()-valid.min())*255).astype(np.uint8)
        base_bgr = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
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
            # r está en coordenadas ABSOLUTAS de la imagen completa; el
            # canvas ahora muestra el RECORTE, así que hay que restar el
            # offset del recorte antes de escalar a píxeles de canvas.
            rx1 = r["x1"] - self._crop_x1
            ry1 = r["y1"] - self._crop_y1
            rx2 = r["x2"] - self._crop_x1
            ry2 = r["y2"] - self._crop_y1
            x1s = int(rx1*self._scale)
            y1s = int(ry1*self._scale)
            x2s = int(rx2*self._scale)
            y2s = int(ry2*self._scale)
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
                f"font-size:15px;font-weight:500;padding:6px;border-radius:6px;"
                f"background:rgb({max(0,c[0]-180)},{max(0,c[1]-180)},{max(0,c[2]-180)});"
                f"color:rgb({c[0]},{c[1]},{c[2]});"
            )
        else:
            self.lbl_instr.setText(
                f"Todas marcadas ({len(self.zone_rects)}) — presiona Guardar"
            )
            self.lbl_instr.setStyleSheet(
                f"font-size:15px;font-weight:500;padding:6px;"
                f"border-radius:6px;background:{TEAL_LIGHT};color:{TEAL};"
            )

        for z, lbl in self._zone_list_labels.items():
            if z in self.zone_rects:
                lbl.setText("✓")
                lbl.setStyleSheet(f"font-size:13px;color:{TEAL};font-weight:500;")
            else:
                lbl.setText("○")
                lbl.setStyleSheet(f"font-size:13px;color:{TEXT_HINT};")

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
            x1 = int(x1s/self._scale) + self._crop_x1
            y1 = int(y1s/self._scale) + self._crop_y1
            x2 = int(x2s/self._scale) + self._crop_x1
            y2 = int(y2s/self._scale) + self._crop_y1
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

        Todo el cuerpo del método va envuelto en try/except: un error acá
        (modelo faltante, imagen inválida, lo que sea) nunca debe cerrar
        la aplicación — a lo sumo, esta ventana se queda sin actualizar.
        """
        try:
            from src.pose_overlay import (
                _detect_landmarks, _landmarks_to_pixels,
                _draw_on_frame, _depth_to_bgr,
                _scaled_parallax, model_available
            )
            import cv2 as _cv2
            from PyQt5.QtGui import QImage, QPixmap

            rgb   = STATE.rgbs.get(self.view_name)
            depth = STATE.depths.get(self.view_name)
            if rgb is None:
                return

            if not model_available():
                QMessageBox.warning(
                    self, "Modelo no disponible",
                    "No se encontró el modelo de detección de landmarks y "
                    "no se pudo descargar automáticamente (revisa tu "
                    "conexión a internet). Puedes colocarlo manualmente en\n"
                    "models/pose_landmarker_full.task"
                )
                return

            # Recortar RGB y depth a la misma sección ROI que muestra el
            # canvas — antes se detectaba sobre la imagen COMPLETA y, si no
            # se conseguían todos los landmarks, terminaba mostrando el
            # depth completo en vez de mantener el recorte delimitado.
            cx1, cy1, cx2, cy2 = self._crop_x1, self._crop_y1, self._crop_x2, self._crop_y2
            rgb_crop   = rgb[cy1:cy2, cx1:cx2]
            depth_crop = depth[cy1:cy2, cx1:cx2] if depth is not None else None

            # Detectar landmarks sobre el RECORTE (en formato RGB para MediaPipe)
            rgb_for_mp = _cv2.cvtColor(rgb_crop.astype("uint8"), _cv2.COLOR_BGR2RGB)
            landmarks  = _detect_landmarks(rgb_for_mp)
            if landmarks is None:
                QMessageBox.information(
                    self, "Sin detección",
                    "No se pudo detectar una persona en la sección "
                    "delimitada de esta vista."
                )
                return

            H_orig, W_orig = rgb_crop.shape[:2]

            # Puntos relativos al RECORTE (sin offset)
            points = _landmarks_to_pixels(landmarks, H_orig, W_orig)

            # Offset de paralaje (normalmente 0 — ver nota en pose_overlay.py;
            # camera.py ya alinea depth↔color por hardware). Se calcula sobre
            # el tamaño de la imagen COMPLETA porque el offset físico no
            # depende de dónde se recorta, y luego se recorta cada punto a
            # los límites del recorte — los puntos que caen fuera
            # simplemente no se dibujan (no se descarta el detector entero).
            off_x, off_y = _scaled_parallax(self._real_w, self._real_h)
            points_depth = []
            for (x, y, vis) in points:
                x_d = x + off_x
                y_d = y + off_y
                if x_d < 0 or x_d >= W_orig or y_d < 0 or y_d >= H_orig:
                    vis = 0.0   # fuera del recorte → no se dibuja
                x_d = max(0, min(W_orig - 1, x_d))
                y_d = max(0, min(H_orig - 1, y_d))
                points_depth.append((x_d, y_d, vis))

            H_canvas, W_canvas = self._img_h, self._img_w

            # ── Actualizar canvas depth (recortado) con landmarks y paralaje ──
            if depth_crop is not None:
                depth_bgr     = _depth_to_bgr(depth_crop)
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

        except Exception as e:
            import traceback
            print(f"⚠ Error detectando landmarks: {e}")
            traceback.print_exc()
            QMessageBox.warning(
                self, "Error detectando landmarks",
                f"Ocurrió un problema al detectar landmarks, pero la "
                f"aplicación sigue funcionando normalmente.\n\nDetalle: {e}"
            )

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
        self._active_view = "frontal"

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(20)

        # ── Centro: visor grande de la vista activa ─────────────────────────
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(text_label("Landmarks y zonas de medida", 24, TEXT_PRI, 700))
        head.addStretch()
        self.seg_views = Segmented([])
        self.seg_views.changed.connect(self._on_view_tab)
        head.addWidget(self.seg_views)
        cv.addLayout(head)

        self.viewer = ImageWithOverlay(dark=True)
        self.viewer.setMinimumHeight(320)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cv.addWidget(self.viewer, stretch=1)

        under = QHBoxLayout(); under.setSpacing(12)
        self.btn_mark = btn("Marcar zonas de esta vista", TEAL, TEAL_LIGHT, icon="target")
        self.btn_mark.clicked.connect(lambda: self._open_zone_marker(self._active_view))
        under.addWidget(self.btn_mark)
        under.addWidget(text_label(
            "Se muestra el mapa de profundidad recortado a la región elegida en Captura.",
            13, TEXT_SEC, 400, wrap=True), stretch=1)
        cv.addLayout(under)
        root.addWidget(center, stretch=1)

        # ── Columna derecha: proceso y medidas ──────────────────────────────
        right = QWidget(); right.setFixedWidth(360)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(14)

        proc, pl = card()
        pl.addWidget(card_title("Proceso", "layers"))

        self.btn_landmarks_p2 = btn("Detectar landmarks (MediaPipe)", icon="target")
        self.btn_landmarks_p2.clicked.connect(self._detect_landmarks_p2)
        pl.addWidget(self.btn_landmarks_p2)

        self.btn_run = btn("Ejecutar pipeline", TEAL, TEAL, variant="solid", icon="play")
        self.btn_run.setMinimumHeight(52)
        self.btn_run.clicked.connect(self._run_pipeline)
        pl.addWidget(self.btn_run)

        self.chk_only_neck = QCheckBox("Solo cuello")
        self.chk_only_neck.setToolTip(
            "Si se activa: la comparación y todos los cálculos derivados "
            "(malla de referencia, volumen por zona, PDF) usan "
            "únicamente la circunferencia del cuello. El resto de zonas, "
            "si se detectan, se ignoran para esos cálculos.\n"
            "Si se deja desactivado, el flujo funciona normal (todas las "
            "zonas disponibles)."
        )
        self.chk_only_neck.setStyleSheet(f"""
            QCheckBox{{font-size:14px;font-weight:600;color:{TEXT_PRI};spacing:8px;
                background:transparent;padding:4px 0;}}
            QCheckBox::indicator{{width:18px;height:18px;border-radius:5px;
                border:1.5px solid {BORDER_STRONG};background:{BG_PANEL};}}
            QCheckBox::indicator:checked{{background:{TEAL};border-color:{TEAL};}}
            QCheckBox::indicator:hover{{border-color:{TEAL};}}
        """)
        self.chk_only_neck.setChecked(STATE.only_neck_mode)
        self.chk_only_neck.toggled.connect(self._on_only_neck_toggled)
        pl.addWidget(self.chk_only_neck)

        self.lbl_only_neck_hint = text_label(
            "Solo se medirá y comparará la circunferencia del cuello.",
            13, TEXT_SEC, 400, wrap=True)
        self.lbl_only_neck_hint.setVisible(STATE.only_neck_mode)
        pl.addWidget(self.lbl_only_neck_hint)
        rv.addWidget(proc)

        meas, ml = card(spacing=0)
        mh = QHBoxLayout()
        mh.addWidget(card_title("Circunferencias", "ruler"))
        self.lbl_meas_count = text_label(f"0 de {len(ZONES)} calculadas", 13, TEXT_SEC, 500)
        mh.addWidget(self.lbl_meas_count)
        ml.addLayout(mh)
        ml.addSpacing(8)

        self._meas_labels = {}
        self._meas_rows   = {}
        for i, zone in enumerate(ZONES):
            row = QFrame(); row.setObjectName(f"mrow_{zone}")
            row.setMinimumHeight(44)
            rl = QHBoxLayout(row); rl.setContentsMargins(10, 0, 10, 0)
            z_lbl = text_label(zone.capitalize(), 14, TEXT_PRI, 500)
            v_lbl = text_label("Pendiente", 14, TEXT_HINT, 400)
            rl.addWidget(z_lbl); rl.addStretch(); rl.addWidget(v_lbl)
            ml.addWidget(row)
            if i < len(ZONES) - 1:
                ml.addWidget(hline())
            self._meas_labels[zone] = v_lbl
            self._meas_rows[zone]   = row
        ml.addSpacing(8)
        self.lbl_height = text_label("Altura estimada: —", 13, TEXT_SEC, 500)
        ml.addWidget(self.lbl_height)
        rv.addWidget(meas)

        self.lbl_status = text_label("Esperando datos...", 13, TEXT_SEC, 400, wrap=True)
        rv.addWidget(self.lbl_status)
        rv.addStretch()

        nav = QHBoxLayout(); nav.setSpacing(12)
        self.btn_prev = btn("Volver", icon="arrow-left")
        self.btn_next = btn("Ver resultados", INK, INK, variant="solid",
                            icon="arrow-right", icon_right=True)
        self.btn_prev.setMinimumHeight(52); self.btn_next.setMinimumHeight(52)
        self.btn_prev.clicked.connect(self.go_prev.emit)
        self.btn_next.clicked.connect(self.go_next.emit)
        self.btn_next.setEnabled(False)
        nav.addWidget(self.btn_prev, stretch=1); nav.addWidget(self.btn_next, stretch=2)
        rv.addLayout(nav)
        root.addWidget(right)

    def showEvent(self, e):
        super().showEvent(e)
        self._rebuild_images()
        self._update_measurements_display()
        if not STATE.pipeline_done:
            self.btn_next.setEnabled(False)

    def _on_view_tab(self, idx):
        names = list(STATE.view_names)
        if 0 <= idx < len(names):
            self._set_active_view(names[idx])

    def _set_active_view(self, name):
        if name not in self._view_images:
            return
        self._active_view = name
        self.seg_views.set_current(list(STATE.view_names).index(name))
        self._refresh_viewer()

    def _on_slot_changed(self, name):
        if name == self._active_view:
            self._refresh_viewer()

    def _refresh_viewer(self):
        slot = (self._view_images.get(self._active_view) or {}).get("depth")
        if slot is not None and slot.array is not None:
            self.viewer.set_image_array(slot.array)
        else:
            self.viewer.clear_image(
                "Sin mapa de profundidad para esta vista.\n"
                "Vuelve a Captura y carga las imágenes."
            )

    def _on_only_neck_toggled(self, checked: bool):
        STATE.only_neck_mode = checked
        self.lbl_only_neck_hint.setVisible(checked)

    def _rebuild_images(self):
        """Vuelve a generar la imagen de cada vista y las pestañas del visor."""
        names = list(STATE.view_names)
        self._view_images = {}
        self.seg_views.set_options([VIEW_LABELS.get(n, n) for n in names])
        self.seg_views.setVisible(len(names) > 1)

        for name in names:
            # Una sola imagen por vista (profundidad recortada a la ROI). Se
            # guarda bajo "rgb" y "depth" por compatibilidad con el resto
            # del código, que escribe en cualquiera de las dos claves.
            slot = _ViewSlot(lambda n=name: self._on_slot_changed(n))
            self._view_images[name] = {"rgb": slot, "depth": slot}
            slot.array = self._render_roi_depth(name)

        if self._active_view not in names:
            self._active_view = names[0]
        self._set_active_view(self._active_view)

    def _roi_bounds(self, name):
        """(x1,y1,x2,y2) del ROI de esta vista, ordenados y recortados a
        los límites reales de la imagen. None si no hay depth cargado."""
        depth = STATE.depths.get(name)
        if depth is None:
            return None
        h, w = depth.shape[:2]
        roi = STATE.roi_params.get(name)
        if not roi:
            return (0, 0, w, h)
        x1, x2 = sorted((roi["x1"], roi["x2"]))
        y1, y2 = sorted((roi["y1"], roi["y2"]))
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return (0, 0, w, h)   # ROI degenerada — usar la imagen completa
        return (x1, y1, x2, y2)

    def _render_roi_depth(self, name):
        """
        Recorta el depth a la zona ROI (x1/x2/y1/y2) fijada en la página
        de Captura y devuelve SOLO esa sección coloreada — no la imagen
        completa con un recuadro encima. Si ya se corrió el pipeline al
        menos una vez, además oscurece dentro del recorte todo lo que
        quedó fuera de la máscara real usada para medir (STATE.segs), así
        que lo que se ve aquí es exactamente lo que se analiza.
        """
        depth = STATE.depths.get(name)
        if depth is None:
            return None
        bounds = self._roi_bounds(name)
        if bounds is None:
            return None
        x1, y1, x2, y2 = bounds
        depth_crop = depth[y1:y2, x1:x2]

        valid = depth_crop[(depth_crop > 300) & (depth_crop < 4000)]
        if len(valid) == 0:
            return None
        d_norm = np.zeros_like(depth_crop, dtype=np.uint8)
        mask_v = (depth_crop > 300) & (depth_crop < 4000)
        d_norm[mask_v] = ((depth_crop[mask_v]-valid.min())/(valid.max()-valid.min())*255).astype(np.uint8)
        img = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)

        seg = STATE.segs.get(name)
        if seg is not None and seg.get("mask") is not None:
            mask_full = seg["mask"]
            if mask_full.shape[:2] == depth.shape[:2]:
                mask_crop = mask_full[y1:y2, x1:x2]
                body_mask = mask_crop > 0
                dimmed = (img.astype(np.float32) * 0.35).astype(np.uint8)
                img = np.where(body_mask[..., None], img, dimmed)

        return img

    # ── MODIFICACIÓN 3 ─────────────────────────────────────────────────────────
    def _update_measurements_display(self):
        """
        Actualiza la lista de circunferencias del panel derecho.

        STATE.measurements ahora es {zona: {"y_px", "circumference_cm", ...}}
        Se mantiene compatibilidad con el formato viejo {zona: float}.
        """
        done = 0
        for zone, lbl in self._meas_labels.items():
            data = STATE.measurements.get(zone)
            if isinstance(data, dict):
                val = data.get("circumference_cm")
            else:
                val = data          # compatibilidad con formato anterior
            if val:
                done += 1
                lbl.setText(f"{val:.1f} cm")
                lbl.setStyleSheet(
                    f"font-family:{FONT_MONO};font-size:16px;font-weight:600;"
                    f"color:{TEXT_PRI};background:transparent;border:none;"
                )
            else:
                lbl.setText("Pendiente")
                lbl.setStyleSheet(
                    f"font-size:14px;font-weight:400;color:{TEXT_HINT};"
                    f"background:transparent;border:none;"
                )
        self.lbl_meas_count.setText(f"{done} de {len(self._meas_labels)} calculadas")
        if STATE.height_cm > 0:
            self.lbl_height.setText(f"Altura estimada: {STATE.height_cm:.1f} cm")
        else:
            self.lbl_height.setText("Altura estimada: —")

    # ── MODIFICACIÓN 4 ─────────────────────────────────────────────────────────
    def _draw_measurement_rects(self):
        """
        Dibuja cuadrantes de medición sobre la imagen depth de cada vista.

        Para cada zona en STATE.measurements lee rect {x1,y1,x2,y2,view}
        y dibuja el rectángulo sobre el widget depth correspondiente.
        """
        COLORS_BGR = {
            "cuello":  (46,  204, 113),
            "pecho":   (52,  152, 219),
            "cintura": (231,  76,  60),
            "cadera":  (155,  89, 182),
            "brazo":   (241, 196,  15),
            "muslo":   (230, 126,  34),
            "rodilla": (52,   73,  94),
        }

        # Agrupar zonas por vista usando all_rects para mostrar en todas
        rects_by_view = {}
        for zona, data in STATE.measurements.items():
            if not isinstance(data, dict):
                continue
            cm        = data.get("circumference_cm")
            all_rects = data.get("all_rects") or {}
            # Fallback a rect simple si no hay all_rects
            if not all_rects and data.get("rect"):
                r = data["rect"]
                all_rects = {r.get("view", "frontal"): r}
            for view, rect in all_rects.items():
                if view not in rects_by_view:
                    rects_by_view[view] = []
                rects_by_view[view].append((zona, rect, cm))

        # Dibujar sobre la imagen depth de cada vista
        for view_name, zone_rects in rects_by_view.items():
            widgets = self._view_images.get(view_name)
            if widgets is None:
                continue

            # Base: depth recortado a la ROI + máscara segmentada (ya
            # oscurece dentro del recorte lo que quedó fuera de la zona
            # x/y/d seleccionada en Captura)
            img = self._render_roi_depth(view_name)
            if img is None:
                continue

            bounds = self._roi_bounds(view_name)
            ox, oy = (bounds[0], bounds[1]) if bounds else (0, 0)
            crop_h, crop_w = img.shape[:2]

            # Dibujar cada cuadrante — los rects de zona vienen en
            # coordenadas de la imagen COMPLETA, pero img ya está
            # recortada a la ROI, así que hay que restar el offset
            # (ox, oy) del recorte y descartar lo que quede fuera.
            for zona, rect, cm in zone_rects:
                color = COLORS_BGR.get(zona, (255, 255, 255))
                x1 = rect["x1"] - ox
                y1 = rect["y1"] - oy
                x2 = rect["x2"] - ox
                y2 = rect["y2"] - oy
                if x2 <= 0 or y2 <= 0 or x1 >= crop_w or y1 >= crop_h:
                    continue   # el rect de esta zona cae fuera del recorte
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(crop_w, x2), min(crop_h, y2)
                cv2.rectangle(img, (x1c, y1c), (x2c, y2c), color, 2)
                label = f"{zona} {cm}cm" if cm else zona
                cv2.putText(img, label, (x1c + 3, max(10, y1c - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                            cv2.LINE_AA)

            # Actualizar widget depth
            w_dep = widgets["depth"] if isinstance(widgets, dict) else widgets
            w_dep.set_image_array(img)

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

        Todo el método va envuelto en try/except (y cada vista se procesa
        en su propio try/except dentro del for): cualquier error nunca
        debe cerrar la aplicación, a lo sumo esa vista queda sin detectar.
        """
        try:
            from src.pose_overlay import draw_landmarks_on_images, model_available

            if not STATE.rgbs:
                self.lbl_status.setText("Carga imágenes primero.")
                return

            if not model_available():
                self.lbl_status.setText(
                    "⚠ El modelo de landmarks no está disponible ni se pudo "
                    "descargar automáticamente — revisa la consola."
                )
                QMessageBox.warning(
                    self, "Modelo no disponible",
                    "No se encontró el modelo de detección de landmarks en "
                    "ninguna de estas rutas:\n"
                    "  - models/pose_landmarker_full.task\n"
                    "  - /tmp/pose_landmarker_full.task\n\n"
                    "Se intentó descargarlo automáticamente y no fue "
                    "posible (revisa tu conexión a internet). También "
                    "puedes colocarlo manualmente en cualquiera de esas "
                    "dos rutas, o revisar la consola para el link de "
                    "descarga manual."
                )
                return

            self.btn_landmarks_p2.setEnabled(False)
            self.lbl_status.setText("Detectando landmarks...")
            QApplication.processEvents()

            ok_views, failed_views = [], []
            for name in STATE.view_names:
                try:
                    rgb   = STATE.rgbs.get(name)
                    depth = STATE.depths.get(name)
                    if rgb is None:
                        continue
                    if depth is None:
                        depth = np.zeros(rgb.shape[:2], dtype=np.float32)

                    # Recortar a la ROI (misma sección que muestra el
                    # preview de esta página, ver _render_roi_depth) antes
                    # de detectar — si no, draw_landmarks_on_images corre
                    # sobre la imagen completa y termina reemplazando el
                    # preview recortado por la imagen completa otra vez.
                    bounds = self._roi_bounds(name)
                    if bounds:
                        x1, y1, x2, y2 = bounds
                        rgb_in   = rgb[y1:y2, x1:x2]
                        depth_in = depth[y1:y2, x1:x2]
                    else:
                        rgb_in, depth_in = rgb, depth

                    rgb_ann, depth_ann, ok = draw_landmarks_on_images(rgb_in, depth_in)

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
                        ok_views.append(name)
                    else:
                        failed_views.append(name)
                except Exception as e:
                    import traceback
                    print(f"⚠ Error detectando landmarks en '{name}': {e}")
                    traceback.print_exc()
                    failed_views.append(name)

            self.btn_landmarks_p2.setEnabled(True)
            if ok_views and not failed_views:
                self.lbl_status.setText(f"Landmarks detectados ✓ ({len(ok_views)}/{len(ok_views)} vistas)")
            elif ok_views and failed_views:
                self.lbl_status.setText(
                    f"Detectado en {len(ok_views)} vista(s): {', '.join(ok_views)}  |  "
                    f"No se detectó en: {', '.join(failed_views)} (ver consola)"
                )
            else:
                self.lbl_status.setText(
                    "No se detectó ninguna persona en ninguna vista. "
                    "Revisa la consola (imprime tamaño/brillo de cada imagen "
                    "analizada) para más detalle."
                )

        except Exception as e:
            import traceback
            print(f"⚠ Error general detectando landmarks: {e}")
            traceback.print_exc()
            self.btn_landmarks_p2.setEnabled(True)
            self.lbl_status.setText("Error detectando landmarks (ver consola).")
            QMessageBox.warning(
                self, "Error detectando landmarks",
                f"Ocurrió un problema, pero la aplicación sigue funcionando "
                f"normalmente.\n\nDetalle: {e}"
            )

    def _run_pipeline(self):
        STATE.measurements.clear()
        STATE.diag.clear()
        STATE.pipeline_done = False
        self.btn_next.setEnabled(False)
        self._update_measurements_display()

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Ejecutando pipeline...")

        if self._thread is not None and self._thread.isRunning():
            self._thread.finished.disconnect()
            self._thread.error.disconnect()
            self._thread.quit()
            self._thread.wait(2000)

        self._thread = PipelineThread()
        self._thread.progress.connect(self.lbl_status.setText)
        self._thread.finished.connect(self._on_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_done(self, _):
        self.btn_run.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.lbl_status.setText("Pipeline completado ✓")
        self._rebuild_images()
        self._update_measurements_display()
        self._draw_measurement_rects()

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
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(20)

        # ── Centro: comparación de mallas + tabla ───────────────────────────
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(text_label("Malla real frente a referencia", 24, TEXT_PRI, 700))
        head.addStretch()
        cv.addLayout(head)

        viewer_card, vl = card(margins=(12, 12, 12, 12), spacing=0)
        self.img_comparison = QLabel(
            "Presiona «Comparar con malla de referencia» para generar\n"
            "y comparar las mallas 3D"
        )
        self.img_comparison.setAlignment(Qt.AlignCenter)
        self.img_comparison.setStyleSheet(
            f"background:{BG_CARD};border:1.5px dashed {BORDER_STRONG};"
            f"border-radius:{RADIUS_SM}px;color:{TEXT_SEC};font-size:15px;"
        )
        self.img_comparison.setMinimumHeight(280)
        vl.addWidget(self.img_comparison)
        cv.addWidget(viewer_card, stretch=3)

        table_card, tcl = card(margins=(20, 16, 20, 16), spacing=10)
        tcl.addWidget(card_title("Tabla de medidas", "ruler"))
        self.table_widget = QWidget()
        self.table_widget.setStyleSheet("background:transparent;")
        tl = QGridLayout(self.table_widget)
        tl.setContentsMargins(0, 0, 0, 0); tl.setSpacing(0)
        headers = ["Zona", "Valor actual", "Valor referencia", "Diferencia"]
        for ci, h in enumerate(headers):
            hl = QLabel(h)
            align = "left" if ci == 0 else "right"
            hl.setAlignment((Qt.AlignLeft if ci == 0 else Qt.AlignRight) | Qt.AlignVCenter)
            hl.setStyleSheet(
                f"background:transparent;color:{TEXT_SEC};padding:8px 10px;"
                f"font-size:13px;font-weight:600;border:none;"
                f"border-bottom:1px solid {BORDER_STRONG};"
            )
            tl.addWidget(hl, 0, ci)

        self._table_rows = {}
        for ri, zone in enumerate(ZONES, 1):
            row_widgets = []
            for ci in range(4):
                lbl = QLabel("—")
                lbl.setAlignment((Qt.AlignLeft if ci == 0 else Qt.AlignRight) | Qt.AlignVCenter)
                fam = "" if ci == 0 else f"font-family:{FONT_MONO};"
                lbl.setStyleSheet(
                    f"background:transparent;padding:7px 10px;font-size:14px;{fam}"
                    f"color:{TEXT_PRI};border:none;border-bottom:1px solid {BORDER};"
                )
                tl.addWidget(lbl, ri, ci)
                row_widgets.append(lbl)
            self._table_rows[zone] = row_widgets
        tcl.addWidget(self.table_widget)
        cv.addWidget(table_card)
        root.addWidget(center, stretch=1)

        # ── Columna derecha: resumen por zona y acciones ────────────────────
        right = QWidget(); right.setFixedWidth(380)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(14)

        top, tpl = card(spacing=6)
        tpl.addWidget(text_label("Mayor diferencia de volumen", 13, TEXT_SEC, 500))
        top_row = QHBoxLayout()
        self.lbl_top_zone = text_label("—", 22, TEXT_PRI, 700)
        self.lbl_top_val  = text_label("—", 30, TEXT_PRI, 600, mono=True)
        top_row.addWidget(self.lbl_top_zone); top_row.addStretch()
        top_row.addWidget(self.lbl_top_val)
        tpl.addLayout(top_row)
        self.lbl_top_sub = text_label(
            "Aún sin comparación con la malla de referencia.", 13, TEXT_SEC, 400, wrap=True)
        tpl.addWidget(self.lbl_top_sub)
        rv.addWidget(top)

        zones_card, zl = card(spacing=12)
        zl.addWidget(card_title("Volumen por zona", "layers"))
        self._zone_cards = {}
        for zone in ["cuello","pecho","cintura","cadera","muslo","rodilla"]:
            cell = QWidget(); cell.setStyleSheet("background:transparent;")
            cl = QVBoxLayout(cell); cl.setContentsMargins(0,0,0,0); cl.setSpacing(5)
            line = QHBoxLayout()
            line.addWidget(text_label(zone.capitalize(), 14, TEXT_PRI, 500))
            line.addStretch()
            lv = text_label("—", 14, TEXT_HINT, 600, mono=True)
            line.addWidget(lv)
            bar = thin_bar(TEAL)
            cl.addLayout(line); cl.addWidget(bar)
            self._zone_cards[zone] = {"val": lv, "bar": bar}
            zl.addWidget(cell)
        zl.addWidget(text_label(
            "Naranja: la malla real supera a la referencia. Azul: queda por debajo. "
            "La barra indica el porcentaje de la zona en exceso.",
            12, TEXT_SEC, 400, wrap=True))
        rv.addWidget(zones_card)

        self.btn_run_smpl = btn("Comparar con malla de referencia", TEAL, TEAL,
                                variant="solid", icon="layers")
        self.btn_run_smpl.setMinimumHeight(52)
        self.btn_run_smpl.clicked.connect(self._run_smpl)
        rv.addWidget(self.btn_run_smpl)

        self.lbl_status = text_label("", 13, TEXT_SEC, 400, wrap=True)
        rv.addWidget(self.lbl_status)
        rv.addStretch()

        nav = QHBoxLayout(); nav.setSpacing(12)
        self.btn_prev = btn("Volver", icon="arrow-left")
        self.btn_prev.clicked.connect(self.go_prev.emit)
        self.btn_pdf = btn("Exportar PDF clínico", TEAL, BG_PANEL, icon="download")
        self.btn_pdf.clicked.connect(self._export_pdf)
        self.btn_prev.setMinimumHeight(52); self.btn_pdf.setMinimumHeight(52)
        nav.addWidget(self.btn_prev, stretch=1); nav.addWidget(self.btn_pdf, stretch=2)
        rv.addLayout(nav)
        root.addWidget(right)

    def showEvent(self, e):
        super().showEvent(e)
        self._update_table()

    def _set_zone_card(self, zone, diff_cm3, pct):
        """Pinta una fila de 'Volumen por zona': valor con signo y barra de color."""
        card_w = self._zone_cards.get(zone)
        if card_w is None:
            return
        if diff_cm3 is None:
            card_w["val"].setText("—")
            card_w["val"].setStyleSheet(
                f"font-family:{FONT_MONO};font-size:14px;font-weight:600;"
                f"color:{TEXT_HINT};background:transparent;border:none;")
            card_w["bar"].setValue(0)
            return
        color = CORAL if diff_cm3 >= 0 else BLUE
        sign  = "+" if diff_cm3 >= 0 else "−"
        card_w["val"].setText(f"{sign}{abs(diff_cm3):.0f} cm³")
        card_w["val"].setStyleSheet(
            f"font-family:{FONT_MONO};font-size:14px;font-weight:600;"
            f"color:{color};background:transparent;border:none;")
        set_bar_color(card_w["bar"], color)
        card_w["bar"].setValue(max(0, min(100, int(pct))))

    def _set_top_card(self, diffs):
        """diffs: {zona: diferencia de volumen en cm³}. Destaca la mayor en valor absoluto."""
        if not diffs:
            self.lbl_top_zone.setText("—")
            self.lbl_top_val.setText("—")
            self.lbl_top_sub.setText("Aún sin comparación con la malla de referencia.")
            return
        zone = max(diffs, key=lambda z: abs(diffs[z]))
        d    = diffs[zone]
        color = CORAL if d >= 0 else BLUE
        sign  = "+" if d >= 0 else "−"
        self.lbl_top_zone.setText(zone.capitalize())
        self.lbl_top_val.setText(f"{sign}{abs(d):.0f} cm³")
        self.lbl_top_val.setStyleSheet(
            f"font-family:{FONT_MONO};font-size:30px;font-weight:600;"
            f"color:{color};background:transparent;border:none;")
        meas = STATE.measurements.get(zone)
        circ = meas.get("circumference_cm") if isinstance(meas, dict) else meas
        if circ:
            self.lbl_top_sub.setText(f"Perímetro medido: {circ:.1f} cm")
        else:
            self.lbl_top_sub.setText(
                "La malla real supera a la referencia." if d >= 0
                else "La malla real queda por debajo de la referencia.")

    def _update_table(self):
        """
        Actualiza la tabla de Page3: valor actual (medido por el
        pipeline, STATE.measurements) y valor de referencia (predicho
        para IMC saludable — solo disponible después de presionar
        'Comparar con Malla de Referencia', STATE.smpl_result).
        """
        # cuello/pecho/cintura/cadera/muslo/rodilla tienen equivalente en
        # el modelo de regresión (predict_measurements); "brazo" no tiene
        # una medida de referencia directa disponible.
        ZONE_TO_REGRESSION_KEY = {
            "cuello": "neck", "pecho": "chest", "cintura": "abdomen",
            "cadera": "hip", "muslo": "thigh", "rodilla": "knee",
        }
        synth_meas = (STATE.smpl_result or {}).get("synth_meas", {})

        for zone, row in self._table_rows.items():
            row[0].setText(zone.capitalize())

            meas_data = STATE.measurements.get(zone)
            actual = (meas_data.get("circumference_cm")
                      if isinstance(meas_data, dict) else meas_data)
            row[1].setText(f"{actual:.1f} cm" if actual else "—")

            # En modo "Solo cuello", la comparación real solo se calculó
            # con el cuello — mostrar referencia de otras zonas aquí
            # confundiría, ya que esos números no participaron en la
            # malla/volumen mostrados arriba.
            if STATE.only_neck_mode and zone != "cuello":
                row[2].setText("—")
                row[3].setText("—")
                continue

            ref_key = ZONE_TO_REGRESSION_KEY.get(zone)
            referencia = synth_meas.get(ref_key) if ref_key else None
            row[2].setText(f"{referencia:.1f} cm" if referencia else "—")

            if actual and referencia:
                diff = actual - referencia
                sign = "+" if diff >= 0 else ""
                row[3].setText(f"{sign}{diff:.1f} cm")
            else:
                row[3].setText("—")

    def _run_smpl(self):
        only_neck = STATE.only_neck_mode

        if only_neck:
            cuello_data = STATE.measurements.get("cuello")
            has_cuello = (isinstance(cuello_data, dict)
                          and cuello_data.get("circumference_cm"))
            if not has_cuello:
                QMessageBox.warning(
                    self, "Falta la medida del cuello",
                    "El modo 'Solo cuello' está activo (página de Mediciones) "
                    "pero todavía no hay una medida de circunferencia del "
                    "cuello. Ejecuta el pipeline en Mediciones primero."
                )
                return
            real_zones = {"cuello": cuello_data}
        else:
            real_zones = {z: d for z, d in STATE.measurements.items()
                          if z in ("cuello", "pecho", "cintura", "cadera")
                          and isinstance(d, dict) and d.get("circumference_cm")}
            if len(real_zones) < 2:
                QMessageBox.warning(
                    self, "Faltan medidas",
                    "Necesito al menos 2 de estas 4 zonas medidas (cuello, pecho, "
                    "cintura, cadera) para generar la malla real. Ejecuta el "
                    "pipeline en la página de Mediciones primero.\n\n"
                    "(Si solo quieres trabajar con el cuello, activa la opción "
                    "'Solo cuello' en la página de Mediciones)."
                )
                return

        self.lbl_status.setText("Generando mallas y comparando...")
        self.btn_run_smpl.setEnabled(False)
        from PyQt5.QtCore import QThread, pyqtSignal as Signal
        class SmplThread(QThread):
            done  = Signal(str)
            error = Signal(str)
            def run(self):
                try:
                    from src.regression_model import UserInputs, predict_measurements
                    from src.volume_comparison import (create_comparison_mesh,
                                                        save_comparison_figure,
                                                        compute_zone_statistics,
                                                        compute_zone_volumes)
                    from src.body_shape_reshaper import model_files_available

                    p = STATE.patient
                    gender = "male" if p["sex"] == "male" else "female"

                    real_fmt = {z: {"circumference_cm": d["circumference_cm"]}
                                for z, d in real_zones.items()}
                    real_target = {z: round(float(v["circumference_cm"]), 1)
                                   for z, v in real_fmt.items()}
                    print(f"  Medidas reales usadas: {real_target}"
                          + (" (modo: SOLO CUELLO)" if only_neck else ""))

                    ideal_weight = round(22.0 * p["height"] ** 2, 1)
                    synth_inputs = UserInputs(
                        body_fat=p["body_fat"], sex=p["sex"], age=p["age"],
                        weight=ideal_weight, height=p["height"],
                    )
                    synth_meas = predict_measurements(synth_inputs)
                    if only_neck:
                        # Modo "solo cuello": la referencia también se
                        # limita a la circunferencia del cuello — todo lo
                        # demás (pecho/cintura/cadera de la malla ideal)
                        # queda en el promedio poblacional del reshaper,
                        # igual que en la malla real. Así la comparación
                        # completa (mallas, volumen por zona, PDF) queda
                        # explicada únicamente por la diferencia en el
                        # cuello, tal como se pidió.
                        synth_fmt = {
                            "cuello": {"circumference_cm": synth_meas["neck"]},
                        }
                    else:
                        synth_fmt = {
                            "cuello":  {"circumference_cm": synth_meas["neck"]},
                            "pecho":   {"circumference_cm": synth_meas["chest"]},
                            "cintura": {"circumference_cm": synth_meas["abdomen"]},
                            "cadera":  {"circumference_cm": synth_meas["hip"]},
                        }
                    print(f"  Peso actual: {p['weight']}kg  →  Peso ideal IMC22: {ideal_weight}kg")

                    if model_files_available(gender):
                        # ── Método preferido: reshaper por medidas
                        # antropométricas (deformación por facetas, ver
                        # src/body_shape_reshaper.py). Genera ambas mallas
                        # con la MISMA topología (12,500 vértices) en un
                        # solve lineal — mucho más rápido que ajustar SMPL
                        # (no hay optimización iterativa) y con mayor
                        # detalle anatómico local.
                        method_label = "reshaper (medidas → malla)"
                        from src.body_shape_reshaper import (
                            BodyShapeReshaper, build_target_mesh,
                            OUR_ZONE_TO_MEASURE,
                        )
                        rs = BodyShapeReshaper(gender)

                        real_mm = {
                            OUR_ZONE_TO_MEASURE[z]: v["circumference_cm"] * 10.0
                            for z, v in real_fmt.items()
                        }
                        real_mm["height"] = p["height"] * 1000.0
                        verts_real, faces = build_target_mesh(real_mm, gender, reshaper=rs)

                        ideal_mm = {
                            OUR_ZONE_TO_MEASURE[z]: v["circumference_cm"] * 10.0
                            for z, v in synth_fmt.items()
                        }
                        ideal_mm["height"] = p["height"] * 1000.0
                        verts_synth, _ = build_target_mesh(ideal_mm, gender, reshaper=rs)
                    else:
                        # ── Respaldo: SMPL + optimización de betas (más
                        # lento, requiere torch/smplx y los archivos .pkl
                        # de SMPL). Se usa solo si no están los archivos
                        # del reshaper en models/body_shape/.
                        method_label = "SMPL (respaldo — reshaper no disponible)"
                        from src.smpl_fitting import (load_smpl_model, generate_smpl_mesh,
                                                       get_vertices)
                        from src.smpl_cache   import load_cached_betas, save_cached_betas
                        smpl_model = load_smpl_model("models", gender)

                        cached_real = load_cached_betas(real_target, gender)
                        if cached_real is not None:
                            verts_real = get_vertices(smpl_model, cached_real)
                        else:
                            verts_real, _, betas_real = generate_smpl_mesh(real_fmt, "models", gender)
                            save_cached_betas(real_target, betas_real, gender)

                        target_flat = {z: round(float(d["circumference_cm"]), 1)
                                       for z, d in synth_fmt.items()}
                        cached_synth = load_cached_betas(target_flat, gender)
                        if cached_synth is not None:
                            verts_synth = get_vertices(smpl_model, cached_synth)
                        else:
                            verts_synth, _, betas_synth = generate_smpl_mesh(synth_fmt, "models", gender)
                            save_cached_betas(target_flat, betas_synth, gender)
                        faces = smpl_model.faces

                    print(f"  Método usado: {method_label}")
                    if only_neck:
                        method_label += " — modo: solo cuello"

                    # Misma topología en ambas mallas → comparación directa
                    # vértice-a-vértice, sin depender de fusión ICP ni de
                    # vecino-más-cercano entre topologías distintas.
                    mesh_real, mesh_synth, signed_dists = create_comparison_mesh(
                        verts_real, verts_synth, faces
                    )
                    save_comparison_figure(
                        verts_real, verts_synth, signed_dists,
                        "output/volume_comparison.png"
                    )
                    zone_stats = compute_zone_statistics(
                        verts_real, verts_synth, signed_dists
                    )
                    # Volumen real en cm³ por zona (corte de malla con
                    # planos horizontales + teorema de la divergencia) —
                    # no una distancia promedio como zone_stats.
                    try:
                        zone_volumes = compute_zone_volumes(mesh_real, mesh_synth)
                    except Exception as e:
                        print(f"  ⚠ No se pudo calcular volumen por zona: {e}")
                        zone_volumes = {}
                    STATE.smpl_result = {
                        "zone_stats":   zone_stats,
                        "zone_volumes": zone_volumes,
                        "img_path":     "output/volume_comparison.png",
                        "real_target":  real_target,
                        "synth_meas":   synth_meas,
                        "ideal_weight": ideal_weight,
                        "method":       method_label,
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
        result  = STATE.smpl_result or {}
        ideal_w = result.get("ideal_weight", "—")
        zonas   = ", ".join(result.get("real_target", {}).keys()) or "—"
        method  = result.get("method", "")
        self.lbl_status.setText(
            f"Completado ✓  |  Peso ideal IMC22: {ideal_w} kg  |  "
            f"Medidas: {zonas}  |  {method}"
        )
        if Path(img_path).exists():
            pix = QPixmap(img_path).scaled(
                self.img_comparison.width(),
                self.img_comparison.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.img_comparison.setPixmap(pix)
        if STATE.smpl_result:
            # Las tarjetas de la izquierda usan los nombres de ZONES
            # (cuello/pecho/brazo/cintura/cadera/muslo/rodilla), pero
            # compute_zone_volumes/compute_zone_statistics usan bandas de
            # altura con otros nombres (cabeza/cuello/pecho/cintura/
            # cadera/muslos/piernas) — no coinciden 1:1. Este mapeo
            # traduce cada tarjeta a su banda correspondiente; "brazo" no
            # tiene una banda de altura equivalente (los brazos no son
            # una banda horizontal del cuerpo), así que esa tarjeta queda
            # sin dato, igual que antes.
            CARD_TO_BAND = {
                "cuello": "cuello", "pecho": "pecho", "cintura": "cintura",
                "cadera": "cadera", "muslo": "muslos", "rodilla": "piernas",
            }
            zone_volumes = STATE.smpl_result.get("zone_volumes", {})
            zone_stats   = STATE.smpl_result.get("zone_stats", {})
            top_diffs = {}
            for zone, card in self._zone_cards.items():
                # En modo "Solo cuello" las otras zonas no participaron
                # en la comparación (quedaron en el promedio poblacional
                # del reshaper) — mostrar un número ahí sugeriría una
                # diferencia medida que en realidad no se calculó.
                if STATE.only_neck_mode and zone != "cuello":
                    self._set_zone_card(zone, None, 0)
                    continue
                band = CARD_TO_BAND.get(zone)
                vol  = zone_volumes.get(band, {}) if band else {}
                st   = zone_stats.get(band, {}) if band else {}
                diff_cm3 = vol.get("vol_diff_cm3")
                pct      = st.get("pct_excess", 50)
                self._set_zone_card(zone, diff_cm3, pct)
                if diff_cm3 is not None:
                    top_diffs[zone] = diff_cm3
            self._set_top_card(top_diffs)

        # La columna "Valor referencia" de la tabla depende de
        # STATE.smpl_result (recién poblado) — refrescar ahora que ya
        # existe.
        self._update_table()

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

    STEPS = ["Captura", "Mediciones", "Resultados"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setMinimumSize(1280, 800)
        self._build()
        self._apply_styles()

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # ── Cabecera de marca con los tres pasos ────────────────────────────
        nav_bar = QFrame()
        nav_bar.setObjectName("navbar")
        nav_bar.setFixedHeight(66)
        nav_bar.setStyleSheet(f"#navbar{{background:{TEAL};border:none;}}")
        nb = QHBoxLayout(nav_bar)
        nb.setContentsMargins(24,0,24,0); nb.setSpacing(10)

        logo = QLabel()
        logo.setPixmap(app_icon_pixmap(40))
        logo.setFixedSize(40, 40)
        logo.setStyleSheet("background:transparent;border:none;")
        nb.addWidget(logo)

        title = QLabel(APP_NAME)
        title.setStyleSheet(
            f"font-size:19px;font-weight:700;color:{TEXT_ON_DARK};"
            f"background:transparent;border:none;margin-right:22px;"
        )
        nb.addWidget(title)

        self._tab_btns = []
        for i, label in enumerate(self.STEPS):
            tb = QPushButton()
            tb.setCheckable(True)
            tb.setFixedHeight(40)
            tb.setCursor(Qt.PointingHandCursor)
            tb.setIconSize(icon_size(16))
            tb.setStyleSheet(f"""
                QPushButton{{background:{TEAL_MID};color:{TEXT_ON_DARK};
                    border:none;border-radius:20px;
                    padding:0 18px;font-size:14px;font-weight:600;}}
                QPushButton:checked{{background:{BG_PANEL};color:{TEAL};font-weight:700;}}
                QPushButton:hover:!checked{{background:#1E7868;}}
            """)
            tb.clicked.connect(lambda _, idx=i: self._go_to(idx))
            nb.addWidget(tb)
            self._tab_btns.append(tb)

        nb.addStretch()
        self.lbl_patient = QLabel("")
        self.lbl_patient.setStyleSheet(
            "font-size:14px;color:#D3ECE5;background:transparent;border:none;"
        )
        nb.addWidget(self.lbl_patient)
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
            if i < idx:
                # Paso ya completado: marca de verificación en vez del número
                tb.setIcon(line_icon("check", MINT, 16, 3.0))
                tb.setText(self.STEPS[i])
            else:
                tb.setIcon(QIcon())
                tb.setText(f"{i + 1}   {self.STEPS[i]}")
        self._update_patient_label()

    def _update_patient_label(self):
        p = STATE.patient
        name = (p.get("name") or "").strip()
        if not name or name == "No especificado":
            name = "sin nombre"
        sex = "F" if p.get("sex") == "female" else "M"
        try:
            age = f"{int(float(p.get('age', 0)))} años"
        except (TypeError, ValueError):
            age = "—"
        self.lbl_patient.setText(f"Paciente: {name}  ·  {sex}  ·  {age}")

    def _apply_styles(self):
        self.setStyleSheet(f"""
            * {{ font-family: {FONT_FAMILY}; }}
            QMainWindow, QDialog {{ background:{BG_MAIN}; }}

            QSlider::groove:horizontal{{height:6px;background:#E6EBEE;border-radius:3px;}}
            QSlider::handle:horizontal{{width:18px;height:18px;margin:-7px 0;
                border-radius:9px;background:{TEAL};border:2px solid {BG_PANEL};}}
            QSlider::handle:horizontal:hover{{background:{TEAL_DARK};}}
            QSlider::sub-page:horizontal{{background:{TEAL};border-radius:3px;}}

            QComboBox{{background:{BG_PANEL};border:1px solid {BORDER_STRONG};
                border-radius:{RADIUS_SM}px;padding:8px 10px;font-size:14px;color:{TEXT_PRI};}}
            QComboBox:hover{{border-color:{TEXT_HINT};}}
            QComboBox:focus{{border-color:{TEAL};}}
            QComboBox::drop-down{{border:none;width:24px;}}
            QComboBox QAbstractItemView{{background:{BG_PANEL};border:1px solid {BORDER};
                border-radius:{RADIUS_SM}px;selection-background-color:{TEAL_LIGHT};
                selection-color:{TEAL_DARK};padding:4px;outline:none;}}

            QLineEdit, QSpinBox{{background:{BG_PANEL};border:1px solid {BORDER_STRONG};
                border-radius:{RADIUS_SM}px;padding:8px 10px;font-size:14px;color:{TEXT_PRI};}}
            QLineEdit:hover, QSpinBox:hover{{border-color:{TEXT_HINT};}}
            QLineEdit:focus, QSpinBox:focus{{border-color:{TEAL};}}

            QScrollArea{{border:none;background:transparent;}}
            QGroupBox{{font-size:14px;}}

            QScrollBar:vertical{{background:transparent;width:10px;margin:2px;}}
            QScrollBar::handle:vertical{{background:{BORDER_STRONG};border-radius:5px;min-height:24px;}}
            QScrollBar::handle:vertical:hover{{background:{TEXT_HINT};}}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical{{height:0;}}
            QScrollBar:horizontal{{background:transparent;height:10px;margin:2px;}}
            QScrollBar::handle:horizontal{{background:{BORDER_STRONG};border-radius:5px;min-width:24px;}}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal{{width:0;}}

            QToolTip{{background:{TEXT_PRI};color:{TEXT_ON_DARK};border:none;
                border-radius:6px;padding:6px 10px;font-size:12px;}}

            QMessageBox{{background:{BG_PANEL};}}
            QMessageBox QLabel{{color:{TEXT_PRI};font-size:14px;}}
            QMessageBox QPushButton{{background:{TEAL};color:{TEXT_ON_DARK};border:none;
                border-radius:{RADIUS_SM}px;padding:8px 18px;font-size:14px;font-weight:600;min-width:72px;}}
            QMessageBox QPushButton:hover{{background:{TEAL_DARK};}}

            QInputDialog QLineEdit{{background:{BG_PANEL};border:1px solid {BORDER_STRONG};
                border-radius:{RADIUS_SM}px;padding:8px 10px;}}
        """)