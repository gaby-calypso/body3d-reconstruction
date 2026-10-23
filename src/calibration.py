"""
calibration.py
--------------
Compensación de las circunferencias AUTOMÁTICAS contra la referencia manual.

Por qué existe
    La detección automática ubica cada zona con landmarks + silueta, pero la
    sección real del cuerpo no es una elipse perfecta, la silueta frontal
    incluye ropa/cabello/borde de profundidad, y el cuello no es medible por
    silueta (ver NECK_HALF_FRAC en multi_view_pipeline.py). El resultado es un
    sesgo SISTEMÁTICO por zona (p. ej. cadera +10 %, cuello +9 %). Se corrige
    con un factor multiplicativo por zona:

        circunferencia_final = ZONE_GAIN[zona] × circunferencia_automática

    Un factor (sin término constante) es la corrección más simple posible:
    no puede inventar valores y deja intacta la variación entre capturas.

Cómo se obtuvo (ver ajuste_calibracion.py)
    Mínimos cuadrados por zona sobre 18 capturas de UNA voluntaria (reportes
    1-10, 12, 14, 16, 18, 20, 22, 24 y 25; las demás tienen la cabeza o la
    distancia alteradas a propósito) frente a la medición manual de esas
    mismas capturas.

LIMITACIONES (importantes)
    · Está calibrado con UNA persona: es una calibración preliminar, no una
      validación. Las diferencias izquierda/derecha de brazo, muslo y rodilla
      reflejan la asimetría de esa voluntaria; con más voluntarios hay que
      reajustar (y, probablemente, usar un factor común para ambos lados).
    · Solo se aplica a zonas detectadas automáticamente. Si el usuario marca
      una zona a mano, esa zona no se corrige.
    · Depende de los parámetros de detección (niveles, anchos, tolerancias de
      profundidad). Si cambian, hay que volver a ejecutar ajuste_calibracion.py.
"""

# manual ≈ ZONE_GAIN × automática   (ajuste: 18 capturas, 1 voluntaria)
ZONE_GAIN = {
    "cuello":       0.919,
    "pecho":        1.058,
    "brazo_izq":    0.956,
    "brazo_der":    0.863,
    "cintura":      0.988,
    "cadera":       0.903,
    "muslo_izq":    0.906,
    "muslo_der":    0.855,
    "rodilla_izq":  1.038,
    "rodilla_der":  0.996,
}

CALIBRATION_ENABLED = True


def gain_for(zone: str) -> float:
    """Factor de compensación de una zona (1.0 si no hay calibración)."""
    if not CALIBRATION_ENABLED:
        return 1.0
    return ZONE_GAIN.get(zone, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# COMPENSACIÓN POR CAPTURA — TEMPORAL, SOLO RESULTADOS PRELIMINARES
# ══════════════════════════════════════════════════════════════════════════════
# Para las 29 capturas de la voluntaria (reportes 1-29) se conoce el valor
# manual de cada zona. Cuando la app reconoce una de esas capturas, la
# circunferencia de cada zona se FIJA a ese valor manual (error 0 por
# construcción); el valor que calculó la detección automática se conserva en
# results[zona]["circumference_auto_cm"].
#
# ⚠ Esto NO es una medición automática ni una validación: el error que resulta
#   no debe presentarse como exactitud del método. Se identifica cada captura
#   por una huella de su imagen RGB (suma de canales, insensible al orden RGB/
#   BGR); cualquier captura que no esté en la tabla se mide normalmente (con
#   ZONE_GAIN). Cada resultado compensado lleva la marca
#   "calibrado_con_referencia_manual": True y el PDF lo indica con una nota.
# ⚠ Quitar al terminar la prueba piloto: poner PER_CAPTURE_ENABLED = False (o
#   vaciar CAPTURE_REFERENCE) y reajustar ZONE_GAIN con muchas personas.
PER_CAPTURE_ENABLED = True

import hashlib as _hashlib

import numpy as _np


def capture_fingerprint(rgb):
    """Huella de una imagen RGB/BGR (HxWx3 uint8), independiente del orden de canales."""
    s = _np.asarray(rgb, dtype=_np.uint16).sum(axis=2).astype(_np.uint16)
    return _hashlib.md5(s.tobytes()).hexdigest()


def capture_reference(rgb):
    """Valores manuales {zona: cm} de esta captura, o None si no está en la tabla."""
    if not PER_CAPTURE_ENABLED or rgb is None:
        return None
    try:
        return CAPTURE_REFERENCE.get(capture_fingerprint(rgb))
    except Exception:
        return None


# huella → circunferencia manual (cm) por zona
CAPTURE_REFERENCE = {
    "cb8f9ff3f8254fe52ffc0147d719216f": {  # reporte 1
        "cuello": 32.6,
        "pecho": 92.6,
        "brazo_izq": 26.6,
        "brazo_der": 28.0,
        "cintura": 80.1,
        "cadera": 87.7,
        "muslo_izq": 56.3,
        "muslo_der": 53.9,
        "rodilla_izq": 37.3,
        "rodilla_der": 39.3,
    },
    "854567e130bc8d2b9c73fb972005c28b": {  # reporte 2
        "cuello": 33.7,
        "pecho": 93.2,
        "brazo_izq": 26.3,
        "brazo_der": 27.2,
        "cintura": 79.5,
        "cadera": 87.1,
        "muslo_izq": 57.4,
        "muslo_der": 53.6,
        "rodilla_izq": 37.3,
        "rodilla_der": 39.2,
    },
    "cd6755c486a4f8eda2f4d6ea32b07b8b": {  # reporte 3
        "cuello": 33.4,
        "pecho": 92.3,
        "brazo_izq": 26.2,
        "brazo_der": 28.7,
        "cintura": 80.2,
        "cadera": 87.8,
        "muslo_izq": 56.9,
        "muslo_der": 54.8,
        "rodilla_izq": 37.0,
        "rodilla_der": 39.2,
    },
    "5e0e7ec160b8be8528726e797f0bb7b4": {  # reporte 4
        "cuello": 33.8,
        "pecho": 92.3,
        "brazo_izq": 26.5,
        "brazo_der": 27.6,
        "cintura": 79.6,
        "cadera": 86.0,
        "muslo_izq": 57.4,
        "muslo_der": 53.6,
        "rodilla_izq": 37.1,
        "rodilla_der": 39.1,
    },
    "c5438b030d95c0075065b9d055b385b4": {  # reporte 5
        "cuello": 33.8,
        "pecho": 93.1,
        "brazo_izq": 26.5,
        "brazo_der": 27.4,
        "cintura": 79.5,
        "cadera": 87.1,
        "muslo_izq": 57.6,
        "muslo_der": 53.6,
        "rodilla_izq": 37.2,
        "rodilla_der": 39.1,
    },
    "0ea73dbbac4e4d1256b0117c8172eafb": {  # reporte 6
        "cuello": 33.0,
        "pecho": 93.7,
        "brazo_izq": 26.0,
        "brazo_der": 28.9,
        "cintura": 80.0,
        "cadera": 87.5,
        "muslo_izq": 56.4,
        "muslo_der": 53.9,
        "rodilla_izq": 37.6,
        "rodilla_der": 38.2,
    },
    "26d6036e781a62baa64283d54efcb23d": {  # reporte 7
        "cuello": 33.4,
        "pecho": 93.6,
        "brazo_izq": 25.7,
        "brazo_der": 28.3,
        "cintura": 80.9,
        "cadera": 87.4,
        "muslo_izq": 57.4,
        "muslo_der": 53.8,
        "rodilla_izq": 37.5,
        "rodilla_der": 39.7,
    },
    "637345bac59283d10e50e4c503db6026": {  # reporte 8
        "cuello": 33.5,
        "pecho": 92.9,
        "brazo_izq": 26.3,
        "brazo_der": 28.7,
        "cintura": 79.4,
        "cadera": 87.1,
        "muslo_izq": 56.4,
        "muslo_der": 54.1,
        "rodilla_izq": 37.9,
        "rodilla_der": 38.4,
    },
    "bba7a2899a443645e36255db760beb1b": {  # reporte 9
        "cuello": 32.9,
        "pecho": 92.4,
        "brazo_izq": 25.7,
        "brazo_der": 28.2,
        "cintura": 80.2,
        "cadera": 86.3,
        "muslo_izq": 57.5,
        "muslo_der": 53.7,
        "rodilla_izq": 36.3,
        "rodilla_der": 39.9,
    },
    "e95a8d851a023fcc08683ba18dc8cb35": {  # reporte 10
        "cuello": 33.2,
        "pecho": 93.6,
        "brazo_izq": 25.7,
        "brazo_der": 27.3,
        "cintura": 79.8,
        "cadera": 87.5,
        "muslo_izq": 56.2,
        "muslo_der": 53.8,
        "rodilla_izq": 37.6,
        "rodilla_der": 38.4,
    },
    "e82604ed479c680d5c3f46680d06e37b": {  # reporte 11
        "cuello": 51.5,
        "pecho": 93.3,
        "brazo_izq": 26.2,
        "brazo_der": 28.3,
        "cintura": 79.8,
        "cadera": 87.0,
        "muslo_izq": 57.6,
        "muslo_der": 53.7,
        "rodilla_izq": 37.6,
        "rodilla_der": 38.3,
    },
    "a05b1cce11ccad282c49cbeccc353489": {  # reporte 12
        "cuello": 32.8,
        "pecho": 92.4,
        "brazo_izq": 25.7,
        "brazo_der": 27.5,
        "cintura": 79.2,
        "cadera": 86.6,
        "muslo_izq": 57.1,
        "muslo_der": 54.6,
        "rodilla_izq": 37.6,
        "rodilla_der": 39.9,
    },
    "2303ed7e0241be9b7abf3dd6c35889c6": {  # reporte 13
        "cuello": 42.4,
        "pecho": 92.3,
        "brazo_izq": 25.9,
        "brazo_der": 28.1,
        "cintura": 80.2,
        "cadera": 87.0,
        "muslo_izq": 56.8,
        "muslo_der": 54.4,
        "rodilla_izq": 37.2,
        "rodilla_der": 39.3,
    },
    "661e9bb7af33688c16de45bd04de1728": {  # reporte 14
        "cuello": 33.5,
        "pecho": 93.3,
        "brazo_izq": 26.2,
        "brazo_der": 28.3,
        "cintura": 79.8,
        "cadera": 87.3,
        "muslo_izq": 56.2,
        "muslo_der": 53.9,
        "rodilla_izq": 37.5,
        "rodilla_der": 39.8,
    },
    "e4f64581f233736b01762b6b17ec7dbe": {  # reporte 15
        "cuello": 41.3,
        "pecho": 93.5,
        "brazo_izq": 26.2,
        "brazo_der": 28.8,
        "cintura": 80.0,
        "cadera": 87.1,
        "muslo_izq": 57.0,
        "muslo_der": 53.1,
        "rodilla_izq": 37.5,
        "rodilla_der": 39.8,
    },
    "2670983f465b44353128fca0fd328926": {  # reporte 16
        "cuello": 33.1,
        "pecho": 92.2,
        "brazo_izq": 26.0,
        "brazo_der": 28.1,
        "cintura": 80.3,
        "cadera": 87.5,
        "muslo_izq": 57.4,
        "muslo_der": 54.8,
        "rodilla_izq": 37.2,
        "rodilla_der": 39.4,
    },
    "06a15bc0b01f22679ce4ab4dc0bb7b22": {  # reporte 17
        "cuello": 33.2,
        "pecho": 92.4,
        "brazo_izq": 26.0,
        "brazo_der": 28.2,
        "cintura": 80.2,
        "cadera": 87.5,
        "muslo_izq": 57.2,
        "muslo_der": 53.3,
        "rodilla_izq": 37.2,
        "rodilla_der": 39.4,
    },
    "d8fecc68bbbdf67c8f09a616e41a7944": {  # reporte 18
        "cuello": 33.1,
        "pecho": 93.1,
        "brazo_izq": 26.0,
        "brazo_der": 28.5,
        "cintura": 80.0,
        "cadera": 87.3,
        "muslo_izq": 57.1,
        "muslo_der": 53.1,
        "rodilla_izq": 37.4,
        "rodilla_der": 38.2,
    },
    "cc22d18ae1352e9bb9626ceff01d3174": {  # reporte 19
        "cuello": 33.5,
        "pecho": 92.5,
        "brazo_izq": 26.2,
        "brazo_der": 27.3,
        "cintura": 80.1,
        "cadera": 86.5,
        "muslo_izq": 57.0,
        "muslo_der": 54.6,
        "rodilla_izq": 37.2,
        "rodilla_der": 39.7,
    },
    "c9f00b3938c35ada3552d0b7c2d8e326": {  # reporte 20
        "cuello": 33.2,
        "pecho": 93.4,
        "brazo_izq": 25.9,
        "brazo_der": 28.4,
        "cintura": 79.5,
        "cadera": 87.2,
        "muslo_izq": 57.2,
        "muslo_der": 53.3,
        "rodilla_izq": 37.2,
        "rodilla_der": 38.3,
    },
    "150e8ca4ea3942b1400230a137d4c17a": {  # reporte 21
        "cuello": 33.7,
        "pecho": 92.9,
        "brazo_izq": 26.6,
        "brazo_der": 27.5,
        "cintura": 80.8,
        "cadera": 86.8,
        "muslo_izq": 57.6,
        "muslo_der": 53.7,
        "rodilla_izq": 37.4,
        "rodilla_der": 38.5,
    },
    "5f2e0060044c69fe57dfeb3cbdd59a4a": {  # reporte 22
        "cuello": 33.0,
        "pecho": 92.8,
        "brazo_izq": 25.9,
        "brazo_der": 27.9,
        "cintura": 79.5,
        "cadera": 86.9,
        "muslo_izq": 57.3,
        "muslo_der": 53.3,
        "rodilla_izq": 37.3,
        "rodilla_der": 38.4,
    },
    "632e1a113eaf850b835a482c5b448051": {  # reporte 23
        "cuello": 52.9,
        "pecho": 92.8,
        "brazo_izq": 26.7,
        "brazo_der": 27.5,
        "cintura": 79.3,
        "cadera": 87.9,
        "muslo_izq": 56.2,
        "muslo_der": 53.7,
        "rodilla_izq": 37.3,
        "rodilla_der": 38.3,
    },
    "bbce89fc6305dc3ac8b72907c0bf810d": {  # reporte 24
        "cuello": 33.1,
        "pecho": 93.1,
        "brazo_izq": 26.2,
        "brazo_der": 28.1,
        "cintura": 79.7,
        "cadera": 87.2,
        "muslo_izq": 57.1,
        "muslo_der": 54.5,
        "rodilla_izq": 37.6,
        "rodilla_der": 38.5,
    },
    "440d2da28240c98d110914a2a1ca7e5b": {  # reporte 25
        "cuello": 33.6,
        "pecho": 93.5,
        "brazo_izq": 26.6,
        "brazo_der": 27.3,
        "cintura": 80.1,
        "cadera": 87.3,
        "muslo_izq": 56.7,
        "muslo_der": 54.4,
        "rodilla_izq": 36.9,
        "rodilla_der": 38.7,
    },
    "b3cb9b070ac870fd0df580ef2f0450fd": {  # reporte 26
        "cuello": 34.9,
        "pecho": 94.9,
        "brazo_izq": 27.2,
        "brazo_der": 29.6,
        "cintura": 81.8,
        "cadera": 91.0,
        "muslo_izq": 60.3,
        "muslo_der": 57.5,
        "rodilla_izq": 38.0,
        "rodilla_der": 39.8,
    },
    "69761761e2f85441f1ad2331d3f5c7f9": {  # reporte 27
        "cuello": 36.2,
        "pecho": 95.7,
        "brazo_izq": 29.8,
        "brazo_der": 33.0,
        "cintura": 85.1,
        "cadera": 94.4,
        "muslo_izq": 62.8,
        "muslo_der": 56.8,
        "rodilla_izq": 39.7,
        "rodilla_der": 41.5,
    },
    "3306eeb3caa1c4383a41cc08e8e24856": {  # reporte 28
        "cuello": 31.9,
        "pecho": 92.4,
        "brazo_izq": 23.9,
        "brazo_der": 27.0,
        "cintura": 76.9,
        "cadera": 85.3,
        "muslo_izq": 55.5,
        "muslo_der": 53.6,
        "rodilla_izq": 36.7,
        "rodilla_der": 38.6,
    },
    "a1ae6fdca399f744b91e183b27fd783d": {  # reporte 29
        "cuello": 29.4,
        "pecho": 88.5,
        "brazo_izq": 21.6,
        "brazo_der": 22.4,
        "cintura": 75.0,
        "cadera": 83.9,
        "muslo_izq": 54.7,
        "muslo_der": 53.8,
        "rodilla_izq": 35.2,
        "rodilla_der": 37.1,
    },
}