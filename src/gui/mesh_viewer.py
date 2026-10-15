"""
mesh_viewer.py
--------------
Visor 3D giratorio de la malla reconstruida para la pantalla de Resultados.

Dibuja UNA sola malla (la real) coloreada con la diferencia respecto a la
malla de referencia: rojo donde la real es mayor, azul donde es menor.
Se gira arrastrando con el ratón o con los botones de vista. No depende de
OpenGL: proyecta y rellena los triángulos con QPainter, así que funciona en
cualquier equipo donde ya corre la aplicación.
"""

import math

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF, QPixmap, QRadialGradient,
    QLinearGradient, QFont, QPainterPath,
)
from PyQt5.QtWidgets import QWidget, QSizePolicy

# Colores del mapa de diferencia (divergente, centrado en gris claro)
_COL_POS = np.array([214, 40, 40], dtype=np.float32)     # rojo: real > referencia
_COL_NEU = np.array([220, 227, 230], dtype=np.float32)   # sin diferencia
_COL_NEG = np.array([63, 143, 224], dtype=np.float32)    # azul: real < referencia

_TXT_SOFT = "#C5D2D8"
_TXT_POS  = "#FF9B94"
_TXT_NEG  = "#9CC8F5"

# Bandas de altura (fracción 0 = pies, 1 = coronilla). Son las mismas que usa
# compute_zone_statistics en src/volume_comparison.py.
ZONE_BANDS = {
    "cuello":  (0.75, 0.85),
    "pecho":   (0.55, 0.75),
    "cintura": (0.40, 0.55),
    "cadera":  (0.25, 0.40),
    "muslos":  (0.10, 0.25),
}
# Lado y altura (fracción) donde se coloca la etiqueta de cada zona
_LABEL_LAYOUT = {
    "cuello":  ("left",  0.93),
    "pecho":   ("right", 0.68),
    "cintura": ("right", 0.50),
    "cadera":  ("right", 0.33),
    "muslos":  ("left",  0.17),
}
_LABEL_NAMES = {"muslos": "Muslo"}


class MeshViewer(QWidget):
    """Malla 3D giratoria con mapa de color, etiquetas por zona y escala."""

    def __init__(self, font_family="Figtree", mono_family="IBM Plex Mono"):
        super().__init__()
        self.setMinimumSize(360, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.OpenHandCursor)
        self._font = font_family
        self._mono = mono_family

        self._verts = None        # (N,3) centrados, y hacia arriba
        self._faces = None        # (F,3) índices
        self._vcol  = None        # (N,3) color por vértice 0-255
        self._vmax_cm = 0.0
        self._zones = {}          # {zona: diferencia media en cm}
        self._yaw = 0.0           # grados; 0 = frontal
        self._cache = None        # QPixmap de la malla para (yaw, tamaño)
        self._cache_key = None
        self._drag_x = None
        self._neck_frac = None
        self._zone_frac = {}
        self._message = ""
        self._fast = False        # calidad reducida mientras se arrastra

        self._spin = QTimer(self)
        self._spin.setInterval(60)
        self._spin.timeout.connect(self._spin_step)

    # ── API ────────────────────────────────────────────────────────────────
    def set_message(self, text):
        """Texto de ayuda que se muestra cuando aún no hay malla."""
        self._message = text
        self.update()

    def has_mesh(self):
        return self._verts is not None

    def clear(self):
        self._verts = self._faces = self._vcol = None
        self._zones = {}
        self._zone_frac = {}
        self._cache = None
        self.set_spinning(False)
        self.update()

    def set_mesh(self, verts, faces, signed_dists, zone_values=None):
        """
        verts:        (N,3) vértices de la malla real (metros, y hacia arriba).
        faces:        (F,3) triángulos.
        signed_dists: (N,) diferencia con la referencia en metros (+ = exceso).
        zone_values:  {zona: diferencia media en cm} para las etiquetas.
        """
        v = np.asarray(verts, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int64)
        d = np.asarray(signed_dists, dtype=np.float32)
        v = v - v.mean(axis=0)

        # Orientar las caras hacia afuera (para poder descartar las traseras)
        tri = v[f]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if float((nrm * tri.mean(axis=1)).sum()) < 0:
            f = f[:, ::-1].copy()

        vmax = float(np.percentile(np.abs(d), 95)) if len(d) else 0.0
        if vmax < 1e-6:
            vmax = 1e-6
        t = np.clip(d / vmax, -1.0, 1.0)[:, None]
        col = np.where(t >= 0,
                       _COL_NEU + (_COL_POS - _COL_NEU) * t,
                       _COL_NEU + (_COL_NEG - _COL_NEU) * (-t))

        self._verts, self._faces, self._vcol = v, f, col.astype(np.float32)
        self._neck_frac = self._find_neck(v)
        self._zone_frac = self._find_zone_heights(v, self._neck_frac)
        self._vmax_cm = vmax * 100.0
        self._zones = dict(zone_values or {})
        self._cache = None
        self.update()

    @staticmethod
    def _find_neck(v):
        """
        Altura real del cuello (fracción 0-1 de la estatura), buscada en la
        propia malla: es el corte horizontal MÁS ESTRECHO entre la cabeza y
        los hombros. Así la etiqueta apunta al cuello aunque la banda fija
        de la zona (75-85 % de la altura) caiga sobre los hombros.
        Devuelve None si no se puede determinar.
        """
        y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
        height = y1 - y0
        if height < 1e-6:
            return None
        frac = (v[:, 1] - y0) / height
        best, best_w = None, None
        for f in np.arange(0.76, 0.925, 0.005):
            m = np.abs(frac - f) < 0.008
            if m.sum() < 6:
                continue
            width = float(v[m, 0].max() - v[m, 0].min())
            if best_w is None or width < best_w:
                best, best_w = float(f), width
        # Si el corte más estrecho sigue siendo muy ancho, no es un cuello
        # (p. ej. malla sin cabeza): se descarta y se usa la banda fija.
        if best is None or best_w > 0.16 * height:
            return None
        return best

    @staticmethod
    def _find_zone_heights(v, neck):
        """
        Altura (fracción 0-1 de la estatura) donde se SEÑALA cada zona,
        buscada en la forma de la propia malla en vez de usar alturas fijas:

          cuello  → corte más estrecho entre cabeza y hombros (_find_neck)
          pecho   → punto más saliente hacia el frente en la parte alta del torso
          cadera  → corte más ancho de la parte baja del torso
          cintura → punto medio entre pecho y cadera
          muslo   → por debajo de la cadera, a media altura del muslo

        Los rangos de búsqueda se expresan respecto a la altura del cuello,
        para que se adapten a las proporciones de cada cuerpo.
        """
        y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
        height = max(y1 - y0, 1e-6)
        frac = (v[:, 1] - y0) / height
        n = neck if neck is not None else 0.85

        def scan(lo, hi, score):
            best, best_s = None, None
            for f in np.arange(lo, hi + 1e-9, 0.005):
                m = np.abs(frac - f) < 0.008
                if m.sum() < 6:
                    continue
                sc = score(v[m])
                if best_s is None or sc > best_s:
                    best, best_s = float(f), sc
            return best

        def torso_width(pts):
            # ancho del tronco: tramo continuo alrededor del eje (sin brazos)
            xs = np.sort(pts[:, 0])
            gap = 0.035 * height
            mid = int(np.searchsorted(xs, 0.0))
            lo_i = min(mid, len(xs) - 1)
            hi_i = lo_i
            while lo_i > 0 and xs[lo_i] - xs[lo_i - 1] < gap:
                lo_i -= 1
            while hi_i < len(xs) - 1 and xs[hi_i + 1] - xs[hi_i] < gap:
                hi_i += 1
            return float(xs[hi_i] - xs[lo_i])

        def front_reach(pts):
            near = pts[np.abs(pts[:, 0]) < 0.14 * height]
            return float(near[:, 2].max()) if len(near) else -1e9

        chest = scan(0.78 * n, 0.86 * n, front_reach) or 0.83 * n
        hip   = scan(0.52 * n, 0.66 * n, torso_width) or 0.60 * n
        waist = (chest + hip) / 2.0
        thigh = hip - 0.13
        out = {"pecho": chest, "cintura": waist, "cadera": hip, "muslos": thigh}
        if neck is not None:
            out["cuello"] = neck
        return out

    def set_yaw(self, degrees):
        self._yaw = float(degrees) % 360.0
        self.update()

    def set_spinning(self, on):
        if on and self._verts is not None:
            self._spin.start()
        else:
            self._spin.stop()

    def is_spinning(self):
        return self._spin.isActive()

    # ── Interacción ────────────────────────────────────────────────────────
    def _spin_step(self):
        self._fast = True
        self.set_yaw(self._yaw + 4.0)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._verts is not None:
            self._drag_x = e.x()
            self._fast = True
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag_x is not None:
            dx = e.x() - self._drag_x
            self._drag_x = e.x()
            self.set_yaw(self._yaw + dx * 0.6)

    def mouseReleaseEvent(self, e):
        if self._drag_x is not None:
            self._drag_x = None
            self.setCursor(Qt.OpenHandCursor)
            if not self._spin.isActive():
                self._fast = False
                self._cache = None
                self.update()

    def resizeEvent(self, e):
        self._cache = None
        super().resizeEvent(e)

    # ── Geometría ──────────────────────────────────────────────────────────
    def _layout(self):
        """Devuelve (cx, cy, escala px/m) para encajar la figura en el widget."""
        w, h = self.width(), self.height()
        v = self._verts
        height_m = float(v[:, 1].max() - v[:, 1].min())
        # Radio horizontal máximo: no cambia al girar, así la figura no "salta"
        radius_m = float(np.sqrt(v[:, 0] ** 2 + v[:, 2] ** 2).max())
        avail_h = h - 90
        avail_w = w - 2 * 215          # margen para etiquetas y escala
        scale = min(avail_h / max(height_m, 1e-6),
                    avail_w / max(2 * radius_m, 1e-6))
        y_mid = float(v[:, 1].max() + v[:, 1].min()) / 2.0
        return w / 2.0, h / 2.0 - 8, scale, y_mid

    def _project(self):
        th = math.radians(self._yaw)
        c, s = math.cos(th), math.sin(th)
        v = self._verts
        xr = v[:, 0] * c + v[:, 2] * s
        zr = -v[:, 0] * s + v[:, 2] * c
        return xr, v[:, 1], zr

    def snapshot(self, yaw, max_height=900):
        """
        Imagen (QPixmap con fondo transparente) de la malla vista desde el
        ángulo `yaw`, recortada a la figura. Se usa para el reporte PDF.
        Todas las vistas salen a la misma escala. None si no hay malla.
        """
        if self._verts is None:
            return None
        old_yaw, old_fast = self._yaw, self._fast
        self._yaw, self._fast = float(yaw), False
        try:
            v = self._verts
            xr, _, _ = self._project()
            height_m = float(v[:, 1].max() - v[:, 1].min())
            scale = (max_height - 12) / max(height_m, 1e-6)
            half = float(np.abs(xr).max()) * scale + 6
            w, h = int(2 * half), int(max_height)
            y_mid = float(v[:, 1].max() + v[:, 1].min()) / 2.0
            return self._render_mesh(w, h, (w / 2.0, h / 2.0, scale, y_mid))
        finally:
            self._yaw, self._fast = old_yaw, old_fast

    def scale_max_cm(self):
        """Valor extremo (±, en cm) de la escala de color actual."""
        return float(self._vmax_cm)

    def _render_mesh(self, w=None, h=None, layout=None):
        """Pinta la malla sombreada en un QPixmap transparente."""
        if w is None or h is None:
            w, h = self.width(), self.height()
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        cx, cy, scale, y_mid = layout if layout is not None else self._layout()
        xr, yr, zr = self._project()
        sx = cx + xr * scale
        sy = cy - (yr - y_mid) * scale

        f = self._faces
        p0 = np.stack([xr[f[:, 0]], yr[f[:, 0]], zr[f[:, 0]]], axis=1)
        p1 = np.stack([xr[f[:, 1]], yr[f[:, 1]], zr[f[:, 1]]], axis=1)
        p2 = np.stack([xr[f[:, 2]], yr[f[:, 2]], zr[f[:, 2]]], axis=1)
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm, axis=1)
        ln[ln < 1e-12] = 1e-12
        nrm /= ln[:, None]

        front = nrm[:, 2] > 0.0
        idx = np.nonzero(front)[0]
        if self._fast and len(idx) > 9000:
            idx = idx[::2]                       # borrador rápido al girar
        depth = (p0[idx, 2] + p1[idx, 2] + p2[idx, 2]) / 3.0
        idx = idx[np.argsort(depth)]             # de atrás hacia adelante

        # Luz principal desde arriba-izquierda-frente + luz ambiente
        light = np.array([-0.35, 0.45, 0.82], dtype=np.float32)
        light /= np.linalg.norm(light)
        shade = 0.42 + 0.58 * np.clip(nrm[idx] @ light, 0.0, 1.0)
        base = self._vcol[f[idx]].mean(axis=1)
        rgb = np.clip(base * shade[:, None], 0, 255).astype(np.int32)

        fx = sx[f[idx]]
        fy = sy[f[idx]]

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, False)
        pen = QPen()
        pen.setWidthF(1.0)
        pen.setJoinStyle(Qt.RoundJoin)
        for i in range(len(idx)):
            col = QColor(int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]))
            pen.setColor(col)
            painter.setPen(pen)                  # tapa las juntas entre caras
            painter.setBrush(col)
            painter.drawPolygon(QPolygonF([
                QPointF(float(fx[i, 0]), float(fy[i, 0])),
                QPointF(float(fx[i, 1]), float(fy[i, 1])),
                QPointF(float(fx[i, 2]), float(fy[i, 2])),
            ]))
        painter.end()
        return pm

    def _zone_anchor(self, zone, side):
        """Punto sobre el contorno del tronco a la altura real de la zona."""
        v = self._verts
        y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
        height = y1 - y0
        f = self._zone_frac.get(zone)
        if f is None:                       # sin detección: centro de la banda
            lo, hi = ZONE_BANDS[zone]
            f = (lo + hi) / 2.0
        mask = np.abs((v[:, 1] - y0) / height - f) < 0.012
        if mask.sum() < 3:
            return None
        cx, cy, scale, y_mid = self._layout()
        xr, yr, _ = self._project()
        xs = np.sort(xr[mask])
        # Recorrer desde el eje hacia afuera y parar en el primer hueco: así
        # se toma el borde del tronco (o de la pierna) y no el de un brazo.
        gap = 0.035 * height
        if side == "left":
            k = max(0, min(int(np.searchsorted(xs, 0.0)), len(xs) - 1))
            if zone == "muslos":            # pierna de ese lado, no el eje
                k = 0
                while k < len(xs) - 1 and xs[k + 1] - xs[k] < gap and xs[k + 1] < 0:
                    k += 1
                x_edge = float(xs[0])
            else:
                while k > 0 and xs[k] - xs[k - 1] < gap:
                    k -= 1
                x_edge = float(xs[k])
        else:
            k = max(0, min(int(np.searchsorted(xs, 0.0)), len(xs) - 1))
            while k < len(xs) - 1 and xs[k + 1] - xs[k] < gap:
                k += 1
            x_edge = float(xs[k])
        inset = 6.0 if side == "left" else -6.0
        y_c = y0 + f * height
        return QPointF(cx + x_edge * scale + inset, cy - (y_c - y_mid) * scale)

    # ── Pintado ────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()

        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, w, h), 16, 16)
        p.setClipPath(clip)
        bg = QRadialGradient(QPointF(w / 2.0, h * 0.38), max(w, h) * 0.75)
        bg.setColorAt(0, QColor("#2C4350"))
        bg.setColorAt(1, QColor("#0F1C22"))
        p.fillRect(QRectF(0, 0, w, h), QBrush(bg))

        if self._verts is None:
            p.setPen(QColor(_TXT_SOFT))
            p.setFont(self._f(15, 500))
            p.drawText(QRectF(40, 0, w - 80, h), Qt.AlignCenter | Qt.TextWordWrap,
                       self._message)
            p.end()
            return

        cx, cy, scale, y_mid = self._layout()
        v = self._verts

        # Sombra en el piso
        foot_y = cy - (float(v[:, 1].min()) - y_mid) * scale
        rad = float(np.sqrt(v[:, 0] ** 2 + v[:, 2] ** 2).max()) * scale * 0.55
        sh = QRadialGradient(QPointF(cx, foot_y), max(rad, 1.0))
        sh.setColorAt(0, QColor(0, 0, 0, 150))
        sh.setColorAt(1, QColor(0, 0, 0, 0))
        p.save()
        p.translate(cx, foot_y)
        p.scale(1.0, 0.16)
        p.translate(-cx, -foot_y)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(sh))
        p.drawEllipse(QPointF(cx, foot_y), rad, rad)
        p.restore()

        key = (round(self._yaw, 2), w, h, self._fast)
        if self._cache is None or self._cache_key != key:
            self._cache = self._render_mesh()
            self._cache_key = key
        p.drawPixmap(0, 0, self._cache)

        self._paint_labels(p, cx, cy, scale, y_mid)
        self._paint_scale(p, w, h)

        p.setPen(QColor(_TXT_SOFT))
        p.setFont(self._f(13, 400))
        p.drawText(QRectF(20, h - 40, w - 40, 24), Qt.AlignLeft | Qt.AlignVCenter,
                   "Arrastra para girar.  Rojo: la malla real es mayor que la "
                   "referencia.  Azul: es menor.")
        p.end()

    def _f(self, px, weight=400, mono=False):
        f = QFont(self._mono if mono else self._font)
        f.setPixelSize(px)
        f.setWeight({400: QFont.Normal, 500: QFont.Medium,
                     600: QFont.DemiBold, 700: QFont.Bold}.get(weight, QFont.Normal))
        if mono:
            f.setStyleHint(QFont.Monospace)
        return f

    def _paint_labels(self, p, cx, cy, scale, y_mid):
        if not self._zones:
            return
        v = self._verts
        y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
        height = y1 - y0
        radius = float(np.sqrt(v[:, 0] ** 2 + v[:, 2] ** 2).max()) * scale
        w = self.width()
        # distancia del eje a la etiqueta; deja libre la escala de color
        gap = max(120.0, min(radius + 36, w / 2.0 - 300))
        top = max(self._zones, key=lambda z: abs(self._zones[z]))

        # Altura de cada etiqueta: la de su zona. Luego se separan las de un
        # mismo lado para que no se encimen (mínimo 76 px entre centros).
        label_y = {}
        for zone in self._zones:
            if zone not in _LABEL_LAYOUT:
                continue
            side, default_frac = _LABEL_LAYOUT[zone]
            f = self._zone_frac.get(zone, default_frac)
            if zone == "cuello":
                f = max(0.90, f + 0.07)           # por encima del brazo
            label_y[zone] = cy - (y0 + f * height - y_mid) * scale
        for side_name in ("left", "right"):
            group = sorted((z for z in label_y if _LABEL_LAYOUT[z][0] == side_name),
                           key=lambda z: label_y[z])
            for prev, cur in zip(group, group[1:]):
                if label_y[cur] - label_y[prev] < 76:
                    label_y[cur] = label_y[prev] + 76

        for zone, value in self._zones.items():
            if zone not in _LABEL_LAYOUT:
                continue
            side = _LABEL_LAYOUT[zone][0]
            anchor = self._zone_anchor(zone, side)
            if anchor is None:
                continue
            ly = label_y[zone]
            lx = cx - gap if side == "left" else cx + gap

            p.setPen(QPen(QColor(255, 255, 255, 210), 1.3))
            p.drawLine(anchor, QPointF(lx + (8 if side == "left" else -8), ly))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#FFFFFF"))
            p.drawEllipse(anchor, 3.5, 3.5)

            name = _LABEL_NAMES.get(zone, zone.capitalize())
            sign = "+" if value > 0.005 else ("−" if value < -0.005 else "")
            txt = f"{sign}{abs(value):.2f} cm"
            highlight = (zone == top)
            box_w = 150
            x_txt = lx - box_w if side == "left" else lx
            align = Qt.AlignRight if side == "left" else Qt.AlignLeft

            if highlight:
                rect = QRectF(x_txt - 12 if side == "right" else x_txt - 2,
                              ly - 34, box_w + 14, 66)
                p.setBrush(QColor("#FFFFFF"))
                p.drawRoundedRect(rect, 10, 10)
                name_col = QColor("#36464E")
                val_col = QColor("#B3261E") if value >= 0 else QColor("#1F5F9E")
                inner = rect.adjusted(12, 0, -12, 0)
                p.setPen(name_col); p.setFont(self._f(14, 500))
                p.drawText(QRectF(inner.x(), ly - 28, inner.width(), 20),
                           align | Qt.AlignVCenter, name)
                p.setPen(val_col); p.setFont(self._f(21, 600, mono=True))
                p.drawText(QRectF(inner.x(), ly - 6, inner.width(), 30),
                           align | Qt.AlignVCenter, txt)
            else:
                if value > 0.005:
                    val_col = QColor(_TXT_POS)
                elif value < -0.005:
                    val_col = QColor(_TXT_NEG)
                else:
                    val_col = QColor("#FFFFFF")
                p.setPen(QColor(_TXT_SOFT)); p.setFont(self._f(14, 500))
                p.drawText(QRectF(x_txt, ly - 26, box_w, 20),
                           align | Qt.AlignVCenter, name)
                p.setPen(val_col); p.setFont(self._f(18, 600, mono=True))
                p.drawText(QRectF(x_txt, ly - 5, box_w, 26),
                           align | Qt.AlignVCenter, txt)

    def _paint_scale(self, p, w, h):
        bar_h = min(300.0, h * 0.5)
        x = w - 96.0
        y = (h - bar_h) / 2.0 - 8
        g = QLinearGradient(0, y, 0, y + bar_h)
        g.setColorAt(0.0, QColor(214, 40, 40))
        g.setColorAt(0.5, QColor(220, 227, 230))
        g.setColorAt(1.0, QColor(63, 143, 224))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(g))
        p.drawRoundedRect(QRectF(x, y, 16, bar_h), 8, 8)
        p.setPen(QColor("#FFFFFF"))
        p.setFont(self._f(13, 500, mono=True))
        v = self._vmax_cm
        p.drawText(QRectF(x + 24, y - 4, 70, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   f"+{v:.1f} cm")
        p.drawText(QRectF(x + 24, y + bar_h / 2 - 9, 70, 18),
                   Qt.AlignLeft | Qt.AlignVCenter, "0")
        p.drawText(QRectF(x + 24, y + bar_h - 14, 70, 18),
                   Qt.AlignLeft | Qt.AlignVCenter, f"−{v:.1f} cm")