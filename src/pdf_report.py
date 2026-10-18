"""
pdf_report.py
-------------
Genera el reporte PDF de SomaRecon 3D en UNA sola página A4.

El reporte es únicamente informativo: muestra los datos capturados, las
medidas y las diferencias respecto a la malla de referencia con tablas y
gráficos. No incluye interpretación clínica, clasificación ni recomendaciones.

Contenido:
    - Encabezado y datos del paciente
    - Vistas de la malla real coloreada con la diferencia respecto a la referencia
    - Circunferencias medidas y de referencia (con barras)
    - Diferencia media de superficie por zona (cm)
    - Diferencia de volumen por zona (cm³)
    - Perfil corporal: radares de circunferencia y de volumen frente a la referencia
    - Notas sobre cómo se obtuvo cada dato

Dependencias: reportlab
    pip3 install reportlab --break-system-packages
"""

from datetime import datetime
from pathlib import Path

# Orden y nombre de las bandas de altura (src/volume_comparison.py)
_BANDS = [
    ("cabeza",  "Cabeza"),
    ("cuello",  "Cuello"),
    ("pecho",   "Pecho"),
    ("cintura", "Cintura"),
    ("cadera",  "Cadera"),
    ("muslos",  "Muslos"),
    ("piernas", "Piernas"),
]
# Circunferencias: (clave en las medidas, clave en la referencia, nombre)
# El brazo no está en el modelo de regresión: su referencia (arm_izq / arm_der)
# sale de la malla de referencia. Muslo y rodilla usan la misma referencia en ambos lados.
_CIRCS = [
    ("cuello",      "neck",    "Cuello"),
    ("pecho",       "chest",   "Pecho"),
    ("brazo_izq",   "arm_izq", "Brazo izq."),
    ("brazo_der",   "arm_der", "Brazo der."),
    ("cintura",     "abdomen", "Cintura"),
    ("cadera",      "hip",     "Cadera"),
    ("muslo_izq",   "thigh",   "Muslo izq."),
    ("muslo_der",   "thigh",   "Muslo der."),
    ("rodilla_izq", "knee",    "Rodilla izq."),
    ("rodilla_der", "knee",    "Rodilla der."),
]
# Si falta la medida por lado, se usa la zona promedio (formato anterior)
_CIRC_FALLBACK = {"brazo_izq": "brazo", "brazo_der": "brazo",
                  "muslo_izq": "muslo", "muslo_der": "muslo",
                  "rodilla_izq": "rodilla", "rodilla_der": "rodilla"}


def _num(value):
    """float(value) o None si no es un número utilizable."""
    try:
        if value is None:
            return None
        f = float(value)
        return f if f == f else None        # descarta NaN
    except (TypeError, ValueError):
        return None


def _circ(meas, key):
    """Circunferencia de la zona `key`; si no existe por lado, la zona promedio."""
    v = _num(meas.get(key))
    if v is None and key in _CIRC_FALLBACK:
        v = _num(meas.get(_CIRC_FALLBACK[key]))
    return v


def _signed(value, decimals=1, unit=""):
    if value is None:
        return "—"
    if abs(value) < 0.5 * 10 ** (-decimals):
        return f"{0:.{decimals}f}{unit}"
    sign = "+" if value > 0 else "–"
    return f"{sign}{abs(value):.{decimals}f}{unit}"


def generate_pdf(img_path: str,
                 zone_cards: dict,
                 output_path: str,
                 patient_data: dict | None = None,
                 meas_real: dict | None = None,
                 synth_meas: dict | None = None,
                 zone_stats: dict | None = None,
                 zone_volumes: dict | None = None,
                 mesh_views: list | None = None,
                 scale_max_cm: float | None = None,
                 study_info: dict | None = None,
                 verts_real=None,
                 verts_ref=None,
                 faces=None) -> None:
    """
    Genera el reporte PDF (una página, solo informativo).

    Args:
        img_path     : imagen de comparación de 4 vistas; solo se usa si no se
                       pasan `mesh_views`.
        zone_cards   : dict zona → objeto con .lbl_val.text() (formato
                       anterior; se usa si no se pasa `zone_stats`).
        output_path  : ruta de salida del PDF.
        patient_data : nombre, sexo, edad, peso, talla y % de grasa.
        meas_real    : circunferencias medidas en cm, por zona.
        synth_meas   : circunferencias de referencia en cm (neck, chest, ...).
        zone_stats   : {banda: {"mean_cm", "pct_excess", "pct_deficit"}}.
        zone_volumes : {banda: {"vol_real_cm3", "vol_ref_cm3", "vol_diff_cm3"}}.
        mesh_views   : [(título, ruta_png), ...] vistas de la malla coloreada.
        scale_max_cm : valor extremo (±) de la escala de color de la malla.
        study_info   : datos del estudio: "views", "method", "ideal_weight",
                       "height_cm", "only_neck".
        verts_real, verts_ref, faces : mallas real y de referencia (misma
                       topología). Se aceptan por compatibilidad; no se usan.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError:
        raise ImportError(
            "reportlab no instalado.\n"
            "Instala con: pip3 install reportlab --break-system-packages"
        )

    # ── Tema ──────────────────────────────────────────────────────────────────
    TEAL   = colors.HexColor("#0B4F43")
    MINT   = colors.HexColor("#8FE3CD")
    INK    = colors.HexColor("#14232A")
    SOFT   = colors.HexColor("#4A5A62")
    LINE   = colors.HexColor("#D5DDE1")
    PANEL  = colors.HexColor("#F3F6F7")
    TRACK  = colors.HexColor("#E3E9EC")
    GREY   = colors.HexColor("#9AA8AF")
    RED    = colors.HexColor("#C62828")
    BLUE   = colors.HexColor("#1F5F9E")
    WHITE  = colors.white
    F, FB  = "Helvetica", "Helvetica-Bold"

    pd_ = patient_data or {}
    meas_real = meas_real or {}
    synth_meas = synth_meas or {}
    zone_volumes = zone_volumes or {}
    info = study_info or {}
    only_neck = bool(info.get("only_neck"))

    # Compatibilidad: reconstruir la diferencia media desde zone_cards
    if not zone_stats:
        zone_stats = {}
        for key, card in (zone_cards or {}).items():
            try:
                txt = card.lbl_val.text()
                val = float(txt.replace("+", "").replace("−", "-").replace("–", "-").replace(" cm", ""))
                zone_stats[key] = {"mean_cm": val}
            except Exception:
                continue

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    PW, PH = A4
    M = 32.0                                  # margen lateral
    CW = PW - 2 * M                           # ancho útil
    c = rl_canvas.Canvas(output_path, pagesize=A4)
    c.setTitle("Reporte de medición corporal 3D")
    c.setAuthor("SomaRecon 3D")

    _SUPER = {"²": "2", "³": "3"}

    def text_width(s, font, size):
        base = "".join(ch for ch in s if ch not in _SUPER)
        sup = sum(1 for ch in s if ch in _SUPER)
        return stringWidth(base, font, size) + sup * stringWidth("3", font, size * 0.62)

    def text(x, y, s, font=F, size=9, color=INK, align="left"):
        """Dibuja texto. Los superíndices ² y ³ se dibujan a mano (más
        pequeños y elevados) porque las fuentes estándar del PDF no siempre
        los incluyen."""
        s = str(s)
        c.setFillColor(color)
        w = text_width(s, font, size)
        if align == "right":
            x -= w
        elif align == "center":
            x -= w / 2
        run = ""
        for ch in s + "\0":
            if ch in _SUPER or ch == "\0":
                if run:
                    c.setFont(font, size)
                    c.drawString(x, y, run)
                    x += stringWidth(run, font, size)
                    run = ""
                if ch in _SUPER:
                    c.setFont(font, size * 0.62)
                    c.drawString(x, y + size * 0.36, _SUPER[ch])
                    x += stringWidth(_SUPER[ch], font, size * 0.62)
            else:
                run += ch

    def fit(s, font, size, max_w):
        """Recorta el texto con «…» si no cabe en max_w."""
        s = str(s)
        if stringWidth(s, font, size) <= max_w:
            return s
        while len(s) > 1 and stringWidth(s + "…", font, size) > max_w:
            s = s[:-1]
        return s + "…"

    def card(x, y_top, w, h, title=None, subtitle=None):
        """Tarjeta con borde fino. Devuelve la y (desde abajo) bajo el título."""
        c.setFillColor(WHITE)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.8)
        c.roundRect(x, y_top - h, w, h, 7, stroke=1, fill=1)
        yy = y_top - 17
        if title:
            text(x + 12, yy, title, FB, 10, INK)
            if subtitle:
                text(x + w - 12, yy, subtitle, F, 7.5, SOFT, "right")
            yy -= 9
        return yy

    # ── Encabezado ────────────────────────────────────────────────────────────
    HEAD = 62.0
    c.setFillColor(TEAL)
    c.rect(0, PH - HEAD, PW, HEAD, stroke=0, fill=1)
    text(M, PH - 27, "SomaRecon 3D", FB, 17, WHITE)
    text(M, PH - 44, "Reporte de medición corporal 3D", F, 10.5, MINT)
    now = datetime.now()
    text(PW - M, PH - 27, now.strftime("%d/%m/%Y"), FB, 12, WHITE, "right")
    text(PW - M, PH - 44, "Fecha del estudio · " + now.strftime("%H:%M"),
         F, 8.5, MINT, "right")

    # ── Datos del paciente ────────────────────────────────────────────────────
    weight = _num(pd_.get("weight"))
    height = _num(pd_.get("height"))
    age    = _num(pd_.get("age"))
    fat    = _num(pd_.get("body_fat"))
    bmi    = weight / height ** 2 if weight and height else None
    ideal  = _num(info.get("ideal_weight"))

    tiles = [
        ("Paciente", str(pd_.get("name") or "No especificado"), 2.3),
        ("Sexo",     str(pd_.get("sex") or "—"), 1.25),
        ("Edad",     f"{age:.0f} años" if age is not None else "—", 1.15),
        ("Peso",     f"{weight:.1f} kg" if weight is not None else "—", 1.0),
        ("Talla",    f"{height:.2f} m" if height is not None else "—", 1.0),
        ("Grasa corporal", f"{fat:.0f} %" if fat is not None else "—", 1.25),
        ("IMC",      f"{bmi:.1f} kg/m²" if bmi is not None else "—", 1.4),
    ]
    gap = 6.0
    total_units = sum(t[2] for t in tiles)
    unit_w = (CW - gap * (len(tiles) - 1)) / total_units
    y_tiles = PH - HEAD - 12
    TILE_H = 40.0
    x = M
    for label, value, units in tiles:
        w = unit_w * units
        c.setFillColor(PANEL)
        c.roundRect(x, y_tiles - TILE_H, w, TILE_H, 6, stroke=0, fill=1)
        text(x + 8, y_tiles - 14, label, F, 7.5, SOFT)
        text(x + 8, y_tiles - 30, fit(value, FB, 10.5, w - 14), FB, 10.5, INK)
        x += w + gap

    # ── Vistas de la malla ────────────────────────────────────────────────────
    y_mesh = y_tiles - TILE_H - 10
    MESH_H = 204.0
    yy = card(M, y_mesh, CW, MESH_H, "Malla real frente a referencia",
              "Color = diferencia de superficie respecto a la malla de referencia")
    area_top = yy - 2
    area_bot = y_mesh - MESH_H + 22
    area_h = area_top - area_bot
    scale_w = 62.0
    area_w = CW - 24 - scale_w

    views = [(t, p) for t, p in (mesh_views or []) if p and Path(p).exists()]

    if views:
        readers = [(t, ImageReader(str(p))) for t, p in views]
        sizes = [r.getSize() for _, r in readers]
        # misma escala para todas las vistas (px → pt)
        k = min(area_h / max(h for _, h in sizes),
                (area_w - 10 * (len(sizes) - 1)) / sum(w for w, _ in sizes))
        total_w = sum(w for w, _ in sizes) * k + 10 * (len(sizes) - 1)
        x = M + 12 + (area_w - total_w) / 2
        for (title, reader), (iw, ih) in zip(readers, sizes):
            w, h = iw * k, ih * k
            c.drawImage(reader, x, area_bot + (area_h - h) / 2, w, h, mask="auto")
            text(x + w / 2, y_mesh - MESH_H + 9, title, F, 8, SOFT, "center")
            x += w + 10
    elif img_path and Path(img_path).exists():
        reader = ImageReader(str(img_path))
        iw, ih = reader.getSize()
        k = min(area_w / iw, (area_h + 12) / ih)
        c.drawImage(reader, M + 12 + (area_w - iw * k) / 2, area_bot - 12,
                    iw * k, ih * k, mask="auto")
    else:
        text(M + 12 + area_w / 2, area_bot + area_h / 2,
             "Sin comparación de mallas en este estudio", F, 9, SOFT, "center")

    # Escala de color
    sx = M + CW - scale_w + 4
    bar_h = min(140.0, area_h - 30)
    by = area_bot + (area_h - bar_h) / 2
    steps = 40
    for i in range(steps):
        t = i / (steps - 1)                    # 0 = abajo (azul) … 1 = arriba (rojo)
        if t >= 0.5:
            u = (t - 0.5) * 2
            col = colors.Color((220 + (214 - 220) * u) / 255,
                               (227 + (40 - 227) * u) / 255,
                               (230 + (40 - 230) * u) / 255)
        else:
            u = (0.5 - t) * 2
            col = colors.Color((220 + (63 - 220) * u) / 255,
                               (227 + (143 - 227) * u) / 255,
                               (230 + (224 - 230) * u) / 255)
        c.setFillColor(col)
        c.rect(sx, by + bar_h * i / steps, 9, bar_h / steps + 0.6, stroke=0, fill=1)
    vmax = _num(scale_max_cm)
    top_lbl = f"+{vmax:.1f} cm" if vmax else "mayor"
    bot_lbl = f"–{vmax:.1f} cm" if vmax else "menor"
    text(sx + 13, by + bar_h - 6, top_lbl, F, 7.5, INK)
    text(sx + 13, by + bar_h / 2 - 3, "0", F, 7.5, INK)
    text(sx + 13, by, bot_lbl, F, 7.5, INK)

    # ── Fila de gráficos 1 ────────────────────────────────────────────────────
    y_r1 = y_mesh - MESH_H - 10
    R1_H = 200.0
    col_gap = 10.0
    col_w = (CW - col_gap) / 2

    # (a) Circunferencias medidas y de referencia
    xa = M
    yy = card(xa, y_r1, col_w, R1_H, "Circunferencias (cm)",
              "medida · referencia · diferencia")
    rows = []
    for key, ref_key, label in _CIRCS:
        m = _circ(meas_real, key)
        r = _num(synth_meas.get(ref_key)) if ref_key else None
        if only_neck and key != "cuello":
            r = None if m is None else r
        rows.append((label, m, r))
    max_val = max([v for _, m, r in rows for v in (m, r) if v] or [1.0])
    row_h = (yy - (y_r1 - R1_H) - 24) / len(rows)
    label_w, num_w = 52.0, 30.0
    bar_x = xa + 12 + label_w
    bar_w = col_w - 24 - label_w - 3 * num_w - 6
    y = yy - 4
    for label, m, r in rows:
        mid = y - row_h / 2
        text(xa + 12, mid - 3, label, F, 8.5, INK)
        for j, (val, col) in enumerate(((m, TEAL), (r, GREY))):
            by2 = mid + 1 - j * 6.5
            c.setFillColor(TRACK)
            c.roundRect(bar_x, by2, bar_w, 4.5, 2, stroke=0, fill=1)
            if val:
                c.setFillColor(col)
                c.roundRect(bar_x, by2, max(3, bar_w * val / max_val), 4.5, 2,
                            stroke=0, fill=1)
        nx = bar_x + bar_w + 6 + num_w
        text(nx, mid - 3, f"{m:.1f}" if m is not None else "—", FB, 8.5, TEAL, "right")
        text(nx + num_w, mid - 3, f"{r:.1f}" if r is not None else "—", F, 8.5, SOFT, "right")
        diff = (m - r) if (m is not None and r is not None) else None
        dcol = INK if diff is None or abs(diff) < 0.05 else (RED if diff > 0 else BLUE)
        text(nx + 2 * num_w, mid - 3, _signed(diff), FB, 8.5, dcol, "right")
        y -= row_h
    ly = y_r1 - R1_H + 10
    c.setFillColor(TEAL); c.rect(xa + 12, ly, 8, 5, stroke=0, fill=1)
    text(xa + 23, ly, "Medida", F, 7.5, SOFT)
    c.setFillColor(GREY); c.rect(xa + 60, ly, 8, 5, stroke=0, fill=1)
    text(xa + 71, ly, "Referencia", F, 7.5, SOFT)

    def diverging(x0, y_top, w, h, title, subtitle, items, fmt):
        """Barras divergentes centradas en 0. items: [(etiqueta, valor|None)]."""
        yy_ = card(x0, y_top, w, h, title, subtitle)
        vals = [abs(v) for _, v in items if v is not None]
        vmax_ = max(vals) if vals else 1.0
        if vmax_ <= 0:
            vmax_ = 1.0
        lab_w, val_w = 44.0, 50.0
        plot_x = x0 + 12 + lab_w
        plot_w = w - 24 - lab_w - val_w
        zero = plot_x + plot_w / 2
        rh = (yy_ - (y_top - h) - 10) / max(1, len(items))
        y_ = yy_ - 4
        c.setStrokeColor(LINE); c.setLineWidth(0.8)
        c.line(zero, yy_ - 2, zero, y_top - h + 8)
        for label, v in items:
            mid = y_ - rh / 2
            text(x0 + 12, mid - 3, label, F, 8.5, INK)
            if v is not None:
                length = (plot_w / 2) * abs(v) / vmax_
                col = RED if v > 0 else BLUE
                if length > 0.4:
                    c.setFillColor(col)
                    if v > 0:
                        c.rect(zero, mid - 4, length, 8, stroke=0, fill=1)
                    else:
                        c.rect(zero - length, mid - 4, length, 8, stroke=0, fill=1)
                tcol = INK if abs(v) < 1e-9 else col
                text(x0 + w - 12, mid - 3, fmt(v), FB, 8.5, tcol, "right")
            else:
                text(x0 + w - 12, mid - 3, "—", F, 8.5, SOFT, "right")
            y_ -= rh

    # (b) Diferencia media de superficie por zona
    xb = M + col_w + col_gap
    items = []
    for key, label in _BANDS:
        v = _num((zone_stats.get(key) or {}).get("mean_cm"))
        if only_neck and key != "cuello":
            v = None
        items.append((label, v))
    diverging(xb, y_r1, col_w, R1_H, "Diferencia de superficie (cm)",
              "real – referencia", items, lambda v: _signed(v, 2, " cm"))

    # ── Fila de gráficos 2 ────────────────────────────────────────────────────
    y_r2 = y_r1 - R1_H - 10
    R2_H = 160.0

    # (c) Diferencia de volumen por zona
    items = []
    for key, label in _BANDS:
        v = _num((zone_volumes.get(key) or {}).get("vol_diff_cm3"))
        if only_neck and key != "cuello":
            v = None
        items.append((label, v))
    diverging(M, y_r2, col_w, R2_H, "Diferencia de volumen (cm³)",
              "real – referencia", items, lambda v: _signed(v, 0, " cm³"))

    # (d) Perfil corporal: dos radares (circunferencia y volumen)
    import math
    yy = card(xb, y_r2, col_w, R2_H, "Perfil corporal", "verde = real · gris punteado = referencia")

    def radar(cx, cy, rad, axes, caption):
        """axes: [(etiqueta, real, referencia)]. Cada eje se escala con su
        propia referencia, de modo que la referencia es un polígono regular."""
        axes = [(l, a, b) for l, a, b in axes if a is not None and b]
        text(cx, y_r2 - R2_H + 8, caption, FB, 7.8, INK, "center")
        n = len(axes)
        if n < 3:
            text(cx, cy, "Sin datos suficientes", F, 7.5, SOFT, "center")
            return
        base = 0.72                                  # radio de la referencia
        def pt(i, k):
            ang = math.pi / 2 - 2 * math.pi * i / n
            return cx + math.cos(ang) * rad * k, cy + math.sin(ang) * rad * k
        def poly(ks, stroke, fill, lw, alpha, dash=None):
            path = c.beginPath()
            for i, k in enumerate(ks):
                x_, y_ = pt(i, k)
                (path.moveTo if i == 0 else path.lineTo)(x_, y_)
            path.close()
            c.saveState()
            c.setStrokeColor(stroke); c.setLineWidth(lw); c.setLineJoin(1)
            if dash:
                c.setDash(*dash)
            if fill is not None:
                c.setFillColor(fill); c.setFillAlpha(alpha)
            c.drawPath(path, stroke=1, fill=0 if fill is None else 1)
            c.restoreState()
        for k in (0.36, 0.72, 1.0):
            poly([k] * n, LINE, None, 0.5, 1)
        c.setStrokeColor(LINE); c.setLineWidth(0.5)
        for i in range(n):
            x_, y_ = pt(i, 1.0)
            c.line(cx, cy, x_, y_)
        poly([base] * n, GREY, GREY, 1.0, 0.22, (2.5, 2))
        ks = [max(0.12, min(1.0, base * a / b)) for _, a, b in axes]
        poly(ks, TEAL, TEAL, 1.5, 0.16)
        c.setFillColor(TEAL)
        for i, k in enumerate(ks):
            x_, y_ = pt(i, k)
            c.circle(x_, y_, 1.6, stroke=0, fill=1)
        for i, (lab, _, _) in enumerate(axes):
            x_, y_ = pt(i, 1.0)
            dx = x_ - cx
            al = "center" if abs(dx) < rad * 0.2 else ("left" if dx > 0 else "right")
            ox = 0 if al == "center" else (3 if dx > 0 else -3)
            oy = 3 if y_ > cy + rad * 0.9 else (-8 if y_ < cy - rad * 0.9 else -2.5)
            text(x_ + ox, y_ + oy, lab, F, 6.8, SOFT, al)

    circ_axes = []
    for key, ref_key, label in _CIRCS:
        m_ = _circ(meas_real, key)
        r_ = _num(synth_meas.get(ref_key)) if ref_key else None
        circ_axes.append((label.replace(" izq.", " I").replace(" der.", " D"), m_, r_))
    vol_axes = []
    for key, label in _BANDS:
        zv = zone_volumes.get(key) or {}
        a_, b_ = _num(zv.get("vol_real_cm3")), _num(zv.get("vol_ref_cm3"))
        if a_ is None and b_ is not None and _num(zv.get("vol_diff_cm3")) is not None:
            a_ = b_ + _num(zv.get("vol_diff_cm3"))
        if only_neck and key != "cuello":
            a_ = None
        vol_axes.append((label, a_, b_))
    rad_ = 44.0
    cy_ = y_r2 - R2_H + 30 + rad_
    radar(xb + col_w * 0.26, cy_, rad_, circ_axes, "Circunferencia")
    radar(xb + col_w * 0.74, cy_, rad_, vol_axes, "Volumen")

    # ── Datos del estudio ─────────────────────────────────────────────────────
    y_n = y_r2 - R2_H - 10
    N_H = y_n - 34
    c.setFillColor(PANEL)
    c.roundRect(M, y_n - N_H, CW, N_H, 7, stroke=0, fill=1)
    text(M + 12, y_n - 15, "Datos del estudio", FB, 9, INK)

    facts = []
    views_n = info.get("views")
    if views_n:
        facts.append(f"Vistas capturadas: {views_n}")
    h_est = _num(info.get("height_cm"))
    if h_est:
        facts.append(f"Altura estimada por la cámara: {h_est:.1f} cm")
    if ideal:
        facts.append(f"Peso usado para la referencia (IMC 22): {ideal:.1f} kg")
    if only_neck:
        facts.append("Modo: solo cuello")
    line = "   ·   ".join(facts)
    text(M + 12, y_n - 28, fit(line, F, 7.8, CW - 24), F, 7.8, SOFT)
    ny = y_n - 39
    method = info.get("method")
    if method:
        text(M + 12, ny, fit(f"Método de malla: {method}", F, 7.8, CW - 24), F, 7.8, SOFT)
        ny -= 11

    notes = [
        "Medida: circunferencia calculada con las imágenes RGB y de profundidad. Referencia: modelo de "
        "regresión para el mismo sexo, edad y talla, con el peso indicado arriba.",
        "Superficie y volumen comparan la malla real con la de referencia por bandas de altura. Perfil "
        "corporal: cada eje es una zona, escalada con su propia referencia; si la línea verde queda fuera "
        "del polígono gris, el valor real es mayor. Rojo: la malla real es mayor. Azul: es menor.",
    ]
    for n in notes:
        words, cur = n.split(), ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if stringWidth(trial, F, 7.3) > CW - 24:
                text(M + 12, ny, cur, F, 7.3, SOFT); ny -= 9.2; cur = wd
            else:
                cur = trial
        if cur:
            text(M + 12, ny, cur, F, 7.3, SOFT); ny -= 9.8

    # ── Pie ───────────────────────────────────────────────────────────────────
    c.setStrokeColor(LINE); c.setLineWidth(0.8)
    c.line(M, 26, PW - M, 26)
    text(M, 15, "Documento informativo generado por SomaRecon 3D. "
               "No contiene interpretación clínica ni diagnóstico.", F, 7.5, SOFT)
    text(PW - M, 15, now.strftime("%d/%m/%Y %H:%M"), F, 7.5, SOFT, "right")

    c.showPage()
    c.save()
    print(f"  ✓ PDF guardado en {output_path}")