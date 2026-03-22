"""
pdf_report.py
-------------
Genera un reporte PDF clínico completo con:
    - Datos del paciente
    - Medidas antropométricas reales vs referencia
    - Comparación volumétrica por zona (tabla + interpretación)
    - Imagen de comparación de mallas
    - Interpretación clínica automática

Dependencias: reportlab
    pip3 install reportlab --break-system-packages
"""

import os
import numpy as np
from datetime import datetime
from pathlib import Path


def generate_pdf(img_path: str,
                 zone_cards: dict,
                 output_path: str,
                 patient_data: dict | None = None,
                 meas_real: dict | None = None,
                 synth_meas: dict | None = None) -> None:
    """
    Genera el reporte PDF clínico.

    Args:
        img_path     : ruta a volume_comparison.png
        zone_cards   : dict zona → ZoneCard (para leer valores)
        output_path  : ruta de salida del PDF
        patient_data : dict con datos del paciente (opcional)
        meas_real    : dict con medidas reales en cm
        synth_meas   : dict con medidas sintéticas predichas
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image,
                                         HRFlowable, KeepTogether)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    except ImportError:
        raise ImportError(
            "reportlab no instalado.\n"
            "Instala con: pip3 install reportlab --break-system-packages"
        )

    # ── Colores del tema ──────────────────────────────────────────────────────
    TEAL    = colors.HexColor("#0F6E56")
    TEAL_LT = colors.HexColor("#E1F5EE")
    BLUE    = colors.HexColor("#1565C0")
    BLUE_LT = colors.HexColor("#E3F2FD")
    RED     = colors.HexColor("#C62828")
    RED_LT  = colors.HexColor("#FFEBEE")
    GRAY    = colors.HexColor("#5F5E5A")
    GRAY_LT = colors.HexColor("#F5F5F3")
    BLACK   = colors.HexColor("#2C2C2A")
    WHITE   = colors.white
    BORDER  = colors.HexColor("#D3D1C7")

    # ── Estilos ───────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "title",
        fontSize=20, fontName="Helvetica-Bold",
        textColor=TEAL, alignment=TA_CENTER,
        spaceAfter=4
    )
    style_subtitle = ParagraphStyle(
        "subtitle",
        fontSize=11, fontName="Helvetica",
        textColor=GRAY, alignment=TA_CENTER,
        spaceAfter=16
    )
    style_section = ParagraphStyle(
        "section",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=TEAL, spaceBefore=14, spaceAfter=6,
        borderPad=4
    )
    style_body = ParagraphStyle(
        "body",
        fontSize=10, fontName="Helvetica",
        textColor=BLACK, leading=14,
        alignment=TA_JUSTIFY, spaceAfter=6
    )
    style_note = ParagraphStyle(
        "note",
        fontSize=9, fontName="Helvetica-Oblique",
        textColor=GRAY, leading=12,
        spaceAfter=4
    )

    # ── Documento ─────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title="Reporte de Composición Corporal 3D",
        author="Body3D Reconstruction System"
    )

    story = []
    W = A4[0] - 4*cm   # ancho disponible

    # ── Encabezado ────────────────────────────────────────────────────────────
    story.append(Paragraph("Reporte de Composición Corporal 3D", style_title))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2,
                              color=TEAL, spaceAfter=12))
    
    # ── Datos del paciente ────────────────────────────────────────────────────
    story.append(Paragraph("Datos del paciente", style_section))

    pd = patient_data or {}
    patient_rows = [
        ["Campo", "Valor"],
        ["Nombre",            pd.get("name",     "No especificado")],
        ["Sexo",              pd.get("sex",      "No especificado")],
        ["Edad",              f"{pd.get('age', '—')} años"],
        ["Peso",              f"{pd.get('weight', '—')} kg"],
        ["Talla",             f"{pd.get('height', '—')} m"],
        ["% Grasa corporal",  f"{pd.get('body_fat', '—')}%"],
        ["Fecha de estudio",  datetime.now().strftime("%d/%m/%Y")],
    ]

    patient_table = Table(patient_rows, colWidths=[6*cm, W - 6*cm])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  TEAL),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("BACKGROUND",   (0, 1), (-1, -1), GRAY_LT),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),  [WHITE, GRAY_LT]),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 10),
        ("TEXTCOLOR",    (0, 1), (-1, -1), BLACK),
        ("FONTNAME",     (0, 1), (0, -1),  "Helvetica-Bold"),
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWHEIGHT",    (0, 0), (-1, -1), 22),
        ("GRID",         (0, 0), (-1, -1), 0.5, BORDER),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # ── Medidas antropométricas ───────────────────────────────────────────────
    story.append(Paragraph("Medidas antropométricas", style_section))
    story.append(Paragraph(
        "Las medidas reales se obtienen del modelo SMPL ajustado a la "
        "silueta capturada. Las medidas de referencia se calculan mediante "
        "un modelo de regresión multivariable a partir del porcentaje de "
        "grasa corporal introducido.",
        style_body
    ))

    zone_labels = {
        "neck":    "Cuello",
        "chest":   "Pecho / Tórax",
        "abdomen": "Abdomen / Cintura",
        "hip":     "Cadera",
        "thigh":   "Muslo",
        "knee":    "Rodilla",
        "wrist":   "Muñeca",
    }
    real_keys = {
        "neck":    "cuello",
        "chest":   "pecho",
        "abdomen": "cintura",
        "hip":     "cadera",
    }

    meas_rows = [["Zona", "Real (cm)", "Referencia (cm)", "Diferencia", "Interpretación"]]
    for key, label in zone_labels.items():
        real_val  = meas_real.get(real_keys.get(key, key), None) \
                    if meas_real else None
        synth_val = synth_meas.get(key, None) if synth_meas else None

        real_str  = f"{real_val:.1f}"  if real_val  is not None else "—"
        synth_str = f"{synth_val:.1f}" if synth_val is not None else "—"

        if real_val is not None and synth_val is not None:
            diff = real_val - synth_val
            diff_str = f"{diff:+.1f}"
            if abs(diff) < 1.0:
                interp = "Normal"
            elif diff > 0:
                interp = "Por encima de referencia"
            else:
                interp = "Por debajo de referencia"
        else:
            diff_str = "—"
            interp   = "—"

        meas_rows.append([label, real_str, synth_str, diff_str, interp])

    col_w = [4.5*cm, 2.5*cm, 3.5*cm, 2.5*cm, W - 13*cm]
    meas_table = Table(meas_rows, colWidths=col_w)

    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  TEAL),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWHEIGHT",     (0, 0), (-1, -1), 20),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR",     (0, 1), (-1, -1), BLACK),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_LT]),
    ])

    # Colorear diferencias
    for i, row in enumerate(meas_rows[1:], start=1):
        diff_str = row[3]
        if diff_str != "—":
            diff_val = float(diff_str)
            if diff_val > 1.0:
                ts.add("BACKGROUND", (3, i), (3, i), RED_LT)
                ts.add("TEXTCOLOR",  (3, i), (3, i), RED)
            elif diff_val < -1.0:
                ts.add("BACKGROUND", (3, i), (3, i), BLUE_LT)
                ts.add("TEXTCOLOR",  (3, i), (3, i), BLUE)

    meas_table.setStyle(ts)
    story.append(meas_table)
    story.append(Spacer(1, 12))

    # ── Imagen de comparación ─────────────────────────────────────────────────
    story.append(Paragraph("Comparación volumétrica 3D", style_section))
    story.append(Paragraph(
        "La siguiente imagen muestra la superposición de la malla corporal "
        "real (coloreada) sobre la referencia sintética (gris). El color "
        "rojo indica zonas con exceso de volumen respecto a la referencia; "
        "el azul indica zonas con déficit de volumen.",
        style_body
    ))

    if Path(img_path).exists():
        img = Image(img_path, width=W, height=W * 0.45)
        story.append(img)
        story.append(Spacer(1, 8))

    # ── Diferencia volumétrica por zona ───────────────────────────────────────
    story.append(Paragraph("Análisis por zona corporal", style_section))

    zone_rows = [["Zona", "Diferencia media (cm)", "% Exceso", "% Déficit",
                  "Estado"]]

    zone_labels_es = {
        "cabeza":  "Cabeza",
        "cuello":  "Cuello",
        "pecho":   "Pecho / Tórax",
        "cintura": "Cintura / Abdomen",
        "cadera":  "Cadera",
        "muslos":  "Muslos",
        "piernas": "Piernas",
    }

    for zone_key, card in zone_cards.items():
        label     = zone_labels_es.get(zone_key, zone_key.capitalize())
        val_text  = card.lbl_val.text()
        try:
            val = float(val_text.replace("+", "").replace(" cm", ""))
        except ValueError:
            val = 0.0

        if abs(val) < 0.5:
            estado = "Normal"
        elif val > 0:
            estado = "Exceso"
        else:
            estado = "Déficit"

        zone_rows.append([
            label,
            val_text,
            "—",
            "—",
            estado,
        ])

    zone_col_w = [4*cm, 3.5*cm, 2.5*cm, 2.5*cm, W - 12.5*cm]
    zone_table = Table(zone_rows, colWidths=zone_col_w)

    zts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  TEAL),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWHEIGHT",     (0, 0), (-1, -1), 20),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR",     (0, 1), (-1, -1), BLACK),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_LT]),
    ])

    for i, row in enumerate(zone_rows[1:], start=1):
        estado = row[4]
        if estado == "Exceso":
            zts.add("BACKGROUND", (4, i), (4, i), RED_LT)
            zts.add("TEXTCOLOR",  (4, i), (4, i), RED)
            zts.add("FONTNAME",   (4, i), (4, i), "Helvetica-Bold")
        elif estado == "Déficit":
            zts.add("BACKGROUND", (4, i), (4, i), BLUE_LT)
            zts.add("TEXTCOLOR",  (4, i), (4, i), BLUE)
            zts.add("FONTNAME",   (4, i), (4, i), "Helvetica-Bold")

    zone_table.setStyle(zts)
    story.append(zone_table)
    story.append(Spacer(1, 12))

    # ── Interpretación clínica ────────────────────────────────────────────────
    story.append(Paragraph("Interpretación clínica", style_section))

    bf  = pd.get("body_fat", None)
    sex = pd.get("sex", "").lower()

    if bf is not None:
        if sex == "male":
            if bf < 6:
                bf_interp = "Porcentaje de grasa esencial. Puede asociarse a riesgos metabólicos."
            elif bf < 14:
                bf_interp = "Rango atlético. Composición corporal óptima para rendimiento deportivo."
            elif bf < 18:
                bf_interp = "Rango fitness. Composición corporal saludable."
            elif bf < 25:
                bf_interp = "Rango aceptable. Sin riesgo metabólico significativo."
            else:
                bf_interp = "Por encima del rango recomendado. Se sugiere evaluación médica."
        else:
            if bf < 14:
                bf_interp = "Porcentaje de grasa muy bajo. Puede asociarse a riesgos hormonales."
            elif bf < 21:
                bf_interp = "Rango atlético. Composición corporal óptima para rendimiento deportivo."
            elif bf < 25:
                bf_interp = "Rango fitness. Composición corporal saludable."
            elif bf < 32:
                bf_interp = "Rango aceptable. Sin riesgo metabólico significativo."
            else:
                bf_interp = "Por encima del rango recomendado. Se sugiere evaluación médica."

        story.append(Paragraph(
            f"<b>Porcentaje de grasa corporal ({bf}%):</b> {bf_interp}",
            style_body
        ))

    story.append(Paragraph(
        "Las diferencias volumétricas mostradas representan la comparación "
        "entre la morfología corporal actual del paciente y una referencia "
        "sintética calculada a partir de sus parámetros antropométricos "
        "esperados. Las zonas en rojo presentan un volumen superior a la "
        "referencia, mientras que las zonas en azul presentan un volumen "
        "inferior. Estas diferencias pueden orientar intervenciones "
        "nutricionales, de actividad física o seguimiento clínico.",
        style_body
    ))

    story.append(Paragraph(
        "Este reporte fue generado de forma automática por el sistema "
        "Body3D Reconstruction y tiene carácter orientativo. No sustituye "
        "la evaluación clínica profesional.",
        style_note
    ))

    story.append(HRFlowable(width="100%", thickness=1,
                              color=BORDER, spaceBefore=12))
    story.append(Paragraph(
        f"Body3D Reconstruction System — {datetime.now().strftime('%Y')}",
        ParagraphStyle("footer", fontSize=8, textColor=GRAY,
                        alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f"  ✓ PDF guardado en {output_path}")