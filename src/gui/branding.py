"""
branding.py
-----------
Identidad visual de SomaRecon 3D: icono de la aplicación (dibujado con
QPainter, sin archivos externos) e iconos de línea para los botones.

El icono combina dos ideas: la figura humana inscrita en un círculo y un
cuadrado (estudio de proporciones) y una línea de escaneo que la divide en
mitad sólida (cuerpo real) y mitad nube de puntos (modelo reconstruido).
"""

import re

from PyQt5.QtCore import Qt, QRectF, QPointF, QByteArray, QSize
from PyQt5.QtGui import (
    QPixmap, QPainter, QPainterPath, QColor, QPen, QBrush,
    QLinearGradient, QIcon,
)

APP_NAME = "SomaRecon 3D"

_BRAND_TEAL = "#0B4F43"
_BRAND_ORANGE = "#E0752A"

# Silueta frontal en un lienzo de 200x480 (mismas coordenadas del diseño).
_FIGURE_PATH = (
    "M100 11 C113 11 122 23 122 38 C122 51 116 60 110 63 L110 74 "
    "C130 78 148 84 154 100 C160 130 166 170 172 215 C174 226 162 230 159 219 "
    "C152 185 144 150 138 128 C136 160 138 180 140 205 C144 250 140 300 134 350 "
    "C132 390 130 430 128 462 C127 472 108 472 108 462 C106 420 104 370 102 300 "
    "L98 300 C96 370 94 420 92 462 C92 472 73 472 72 462 C70 430 68 390 66 350 "
    "C60 300 56 250 60 205 C62 180 64 160 62 128 C56 150 48 185 41 219 "
    "C38 230 26 226 28 215 C34 170 40 130 46 100 C52 84 70 78 90 74 L90 63 "
    "C84 60 78 51 78 38 C78 23 87 11 100 11 Z"
)


def _parse_path(d: str) -> QPainterPath:
    """Convierte un path SVG con comandos absolutos M / L / C / Z."""
    path = QPainterPath()
    for cmd, args in re.findall(r"([MLCZ])([^MLCZ]*)", d):
        n = [float(v) for v in args.replace(",", " ").split()]
        if cmd == "M":
            path.moveTo(n[0], n[1])
        elif cmd == "L":
            for i in range(0, len(n), 2):
                path.lineTo(n[i], n[i + 1])
        elif cmd == "C":
            for i in range(0, len(n), 6):
                path.cubicTo(n[i], n[i + 1], n[i + 2], n[i + 3], n[i + 4], n[i + 5])
        elif cmd == "Z":
            path.closeSubpath()
    return path


def figure_path() -> QPainterPath:
    """Silueta humana en coordenadas 200x480, para reutilizar en la interfaz."""
    return _parse_path(_FIGURE_PATH)


def app_icon_pixmap(size: int = 256) -> QPixmap:
    """Dibuja el icono de la aplicación (versión de fondo claro)."""
    scale = 4 if size <= 64 else 2          # sobremuestreo para bordes suaves
    px = size * scale
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.scale(px / 200.0, px / 200.0)         # se dibuja en un lienzo de 200x200

    tile = QPainterPath()
    tile.addRoundedRect(QRectF(0, 0, 200, 200), 46, 46)
    p.setClipPath(tile)

    bg = QLinearGradient(0, 0, 200, 200)
    bg.setColorAt(0, QColor("#FBFDFD"))
    bg.setColorAt(1, QColor("#D5E0E4"))
    p.fillRect(QRectF(0, 0, 200, 200), QBrush(bg))

    teal = QColor(_BRAND_TEAL)

    # Cuadrado de proporciones
    sq = QColor(teal)
    sq.setAlphaF(0.4)
    p.setPen(QPen(sq, 1.5))
    p.setBrush(Qt.NoBrush)
    p.drawRect(QRectF(34, 34, 132, 132))

    # Círculo: mitad izquierda continua, mitad derecha punteada
    circle = QRectF(16, 16, 168, 168)
    p.setPen(QPen(teal, 3.5))
    p.drawArc(circle, 90 * 16, 180 * 16)
    dotted = QPen(teal, 4.0, Qt.CustomDashLine, Qt.RoundCap)
    dotted.setDashPattern([0.03, 2.25])
    p.setPen(dotted)
    p.drawArc(circle, 90 * 16, -180 * 16)

    # Figura: mitad sólida + mitad nube de puntos
    p.save()
    p.translate(64, 14)
    p.scale(0.36, 0.36)
    fig = figure_path()
    p.setClipPath(fig, Qt.IntersectClip)
    body = QLinearGradient(0, 0, 0, 480)
    body.setColorAt(0, QColor("#14967F"))
    body.setColorAt(1, QColor("#0A3340"))
    p.fillRect(QRectF(0, 0, 100, 480), QBrush(body))
    p.setPen(Qt.NoPen)
    p.setBrush(teal)
    y = 11.0
    while y < 480:
        x = 111.0
        while x < 200:
            p.drawEllipse(QPointF(x, y), 5.5, 5.5)
            x += 22
        y += 22
    p.restore()

    # Línea de escaneo
    glow = QColor(_BRAND_ORANGE)
    glow.setAlphaF(0.14)
    p.fillRect(QRectF(91, 0, 18, 200), glow)
    p.fillRect(QRectF(98.5, 0, 3, 200), QColor(_BRAND_ORANGE))
    p.end()

    if scale != 1:
        pm = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm


def app_icon() -> QIcon:
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(app_icon_pixmap(s))
    return icon


# ── Iconos de línea ────────────────────────────────────────────────────────────
_LINE_ICONS = {
    "camera":  '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "upload":  '<path d="M12 16V4M7 9l5-5 5 5M4 20h16"/>',
    "stop":    '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    "play":    '<path d="M7 5l12 7-12 7z"/>',
    "download": '<path d="M12 3v12M7 10l5 5 5-5M4 21h16"/>',
    "arrow-right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "arrow-left":  '<path d="M19 12H5M11 6l-6 6 6 6"/>',
    "check":   '<path d="M5 12l4 4 10-10"/>',
    "user":    '<circle cx="12" cy="8" r="4"/><path d="M4 21c1-4 4-6 8-6s7 2 8 6"/>',
    "target":  '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2.5"/>'
               '<path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>',
    "layers":  '<path d="M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17.5l9 5 9-5"/>',
    "ruler":   '<path d="M3 17L17 3l4 4L7 21zM8 12l2 2M11 9l2 2M14 6l2 2"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/>'
               '<circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/>'
               '<circle cx="18" cy="18" r="2"/>',
}


def line_icon(name: str, color: str = "#14232A", size: int = 20,
              stroke: float = 2.0) -> QIcon:
    """Icono de línea (trazo) del color indicado. QIcon vacío si falta QtSvg."""
    body = _LINE_ICONS.get(name)
    if body is None:
        return QIcon()
    try:
        from PyQt5.QtSvg import QSvgRenderer
    except Exception:
        return QIcon()
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
           f'fill="none" stroke="{color}" stroke-width="{stroke}" '
           f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    icon = QIcon(pm)
    return icon


def icon_size(size: int = 20) -> QSize:
    return QSize(size, size)
