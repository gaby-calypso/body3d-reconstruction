"""
body_shape_reshaper.py
-----------------------
Genera una malla 3D humana a partir de medidas antropométricas, portando
la lógica de inferencia del repo "3D-Human-Body-Shape" (Zeng, Fu, Chao —
ICIMCS 2017, https://github.com/zengyh1900/3D-Human-Body-Shape) a este
proyecto — sin sus dependencias pesadas (mayavi, pyqt5, cvxpy, fancyimpute).

Cómo funciona el modelo (resumen del paper):
    El cuerpo se representa por DEFORMACIÓN de cada triángulo (facet) de
    una malla plantilla, no por posiciones de vértices directamente (esto
    es más expresivo que un espacio PCA de vértices tipo SMPL — captura
    variación local por zona del cuerpo). Para cada facet, un modelo de
    regresión lineal LOCAL (entrenado offline sobre el dataset SPRING,
    seleccionando automáticamente qué medidas antropométricas son
    relevantes para esa zona — "feature-selection-based local mapping")
    predice la deformación de ese facet a partir del vector de 19 medidas.
    Con las 25,000 deformaciones predichas, se resuelve un sistema lineal
    (mínimos cuadrados disperso) que encuentra los vértices consistentes
    con todas ellas — esto es lo que hace d_synthesize().

Qué se reutiliza de ese repo y qué no:
    SÍ: las matrices YA ENTRENADAS en release_model/ (rfemat, rfemask,
        mean_measure, std_measure, d2v) — estas SÍ vienen incluidas en su
        repositorio de GitHub bajo licencia MIT, no requieren el dataset
        SPRING para usarse (solo para reentrenar).
    NO: el dataset SPRING en sí — no hace falta para generar mallas con
        medidas nuevas, solo sería necesario si se quisiera reentrenar el
        modelo desde cero con datos propios.

Medidas esperadas (M_STR, orden exacto — ver utils.py del repo original):
    0  weight                        (no calibrado en este puerto — se
                                       deja siempre en el promedio
                                       poblacional, ver nota más abajo)
    1  height                        mm
    2  neck                          mm (circunferencia)
    3  chest                         mm (circunferencia)
    4  belly button waist            mm (circunferencia — equivale a
                                       nuestra "cintura")
    5  gluteal hip                   mm (circunferencia — equivale a
                                       nuestra "cadera")
    6-18  resto de medidas (longitudes de brazo, muñeca, rodilla, etc.)
                                       — no las medimos actualmente, se
                                       dejan en el promedio poblacional.

Nota sobre "weight": el archivo mean_measure.npy del modelo original
reporta un valor (~4093) que no corresponde a kilogramos reales bajo
ninguna conversión de unidades verificable sin acceso al preprocesamiento
original del dataset SPRING. Verificamos empíricamente que "height" SÍ
está en mm correctos (1631.98mm de la plantilla promedio coincide con el
tamaño real de la malla generada), así que confiamos en las medidas de
longitud/circunferencia, pero dejamos "weight" siempre en el promedio
poblacional (entrada estandarizada = 0) para no inyectar un valor con
una escala no verificada. Si se llega a calibrar ese factor más adelante,
basta con agregarlo a MEASURE_INDEX y pasar el valor real.

Cualquier medida no provista se deja en el promedio poblacional (entrada
estandarizada = 0) — es una simplificación razonable del paso de
imputación MICE del paper original (que requiere fancyimpute + el
dataset de entrenami31do completo, ninguno de los dos disponible aquí).
"""

from __future__ import annotations
import os
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# ── Constantes del modelo (ver utils.py del repo original) ───────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "body_shape")
M_NUM = 19
F_NUM = 25000
V_NUM = 12500

M_STR = ["weight", "height", "neck", "chest",
         "belly button waist", "gluteal hip",
         "neck shoulder elbow wrist", "crotch knee floor",
         "across back shoulder neck", "neck to gluteal hip",
         "natural waist", "max. hip", "natural waist rise",
         "shoulder to midhand", "upper arm", "wrist",
         "outer natural waist to floor", "knee", "max. thigh"]

# Índice en el vector de 19 medidas para cada nombre de M_STR
MEASURE_INDEX = {name: i for i, name in enumerate(M_STR)}

# Mapeo de nuestras zonas medidas (STATE.measurements) a las del modelo.
# "cintura" del pipeline se mide a la altura del ombligo → corresponde a
# "belly button waist", no a "natural waist" (son dos puntos anatómicos
# distintos en el paper original, no se deben mezclar).
OUR_ZONE_TO_MEASURE = {
    "cuello":  "neck",
    "pecho":   "chest",
    "cintura": "belly button waist",
    "cadera":  "gluteal hip",
}


def model_files_available(gender: str = "female") -> bool:
    """True si los archivos del modelo (release_model portado) están presentes."""
    label = "male" if gender == "male" else "female"
    needed = [f"{label}_rfemask.npy", f"{label}_rfemat.npy",
              f"{label}_mean_measure.npy", f"{label}_std_measure.npy",
              f"{label}_d2v.npz", "facets.npy", "normals.npy"]
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in needed)


class BodyShapeReshaper:
    """
    Envoltorio ligero (solo numpy/scipy) del método "local_with_rfemat"
    del repo 3D-Human-Body-Shape — el que el paper reporta como más preciso
    de sus 3 variantes (mapeo global / local con máscara / local con
    selección de features vía RFE).
    """

    def __init__(self, gender: str = "female"):
        label = "male" if gender == "male" else "female"
        self.label = label

        def _load(name):
            return np.load(os.path.join(MODEL_DIR, name), allow_pickle=True)

        self.facets       = _load("facets.npy")            # (F_NUM, 3) int, 1-indexado
        self.rfemask       = _load(f"{label}_rfemask.npy")  # (M_NUM, F_NUM) bool
        self.rfemat         = _load(f"{label}_rfemat.npy")   # (F_NUM,) de matrices
        self.mean_measure   = _load(f"{label}_mean_measure.npy")  # (M_NUM, 1)
        self.std_measure    = _load(f"{label}_std_measure.npy")   # (M_NUM, 1)

        loader = np.load(os.path.join(MODEL_DIR, f"{label}_d2v.npz"))
        d2v = scipy.sparse.coo_matrix(
            (loader["data"], (loader["row"], loader["col"])), shape=loader["shape"]
        )
        self.d2v = d2v
        self._lu = scipy.sparse.linalg.splu(d2v.transpose().dot(d2v).tocsc())

    def generate(self, standardized_measure: np.ndarray) -> np.ndarray:
        """
        Genera vértices (V_NUM, 3) — en METROS, eje Z = altura (ver nota
        de ejes más abajo — build_target_mesh() ya los reordena a Y-up
        antes de devolverlos, para que encajen con el resto del proyecto.

        Args:
            standardized_measure: (M_NUM, 1) — cada medida ya estandarizada
                ((valor - media) / std). Usar build_measure_vector() para
                construir esto a partir de medidas reales.
        """
        w = np.asarray(standardized_measure, dtype=np.float64).reshape(M_NUM, 1)
        w = w * self.std_measure + self.mean_measure

        d = np.empty((F_NUM, 9), dtype=np.float64)
        for i in range(F_NUM):
            mask = np.asarray(self.rfemask[:, i]).reshape(M_NUM, 1)
            alpha = w[mask].reshape(-1, 1)
            d[i, :] = self.rfemat[i].dot(alpha).ravel()

        d_flat = d.reshape(F_NUM * 9, 1)
        Atd = self.d2v.transpose().dot(d_flat)
        x = self._lu.solve(Atd)
        x = x[:V_NUM * 3].reshape(V_NUM, 3)
        x -= x.mean(axis=0)
        return x


def build_measure_vector(known_mm: dict, gender: str = "female") -> np.ndarray:
    """
    Construye el vector estandarizado de 19 medidas que espera el modelo.

    Args:
        known_mm: dict {nombre_de_M_STR: valor_en_mm} con las medidas que
            sí tenemos (ver MEASURE_INDEX para los nombres válidos). Las
            que no se incluyan quedan en el promedio poblacional.
        gender:   "male" o "female" — cada género tiene su propia
            normalización (mean_measure/std_measure)

    Returns:
        (M_NUM, 1) vector estandarizado, listo para BodyShapeReshaper.generate()
    """
    label = "male" if gender == "male" else "female"
    mean_measure = np.load(os.path.join(MODEL_DIR, f"{label}_mean_measure.npy"))
    std_measure  = np.load(os.path.join(MODEL_DIR, f"{label}_std_measure.npy"))

    raw = mean_measure.copy()  # por defecto: promedio poblacional (estandarizado = 0)
    for name, value_mm in known_mm.items():
        if name not in MEASURE_INDEX:
            raise ValueError(f"Medida desconocida: '{name}'. "
                              f"Válidas: {list(MEASURE_INDEX)}")
        raw[MEASURE_INDEX[name], 0] = value_mm

    standardized = (raw - mean_measure) / std_measure
    return standardized


def build_target_mesh(known_mm: dict, gender: str = "female",
                       reshaper: "BodyShapeReshaper | None" = None) -> tuple:
    """
    Función de conveniencia: medidas → (vertices Y-up en metros, facets).

    Reordena los ejes de salida del modelo (X, Y_orig, Z_orig=altura) a
    convención Y-up (X, Z_orig, Y_orig) para que encajen directamente con
    el resto de este proyecto (align_meshes/compute_vertex_distances/etc.
    en src/volume_comparison.py ya asumen Y como eje vertical).

    Args:
        known_mm: ver build_measure_vector()
        gender:   "male" o "female"
        reshaper: instancia ya cargada (evita recargar las matrices — 60MB
            — si se van a generar varias mallas seguidas, p.ej. real+ideal)

    Returns:
        (vertices (V_NUM,3) float64 en metros Y-up, facets (F_NUM,3) int
         base-0, listos para usar directamente como índices de numpy /
         Open3D)
    """
    if reshaper is None:
        reshaper = BodyShapeReshaper(gender)
    std_vec = build_measure_vector(known_mm, gender)
    verts_raw = reshaper.generate(std_vec)
    verts_yup = verts_raw[:, [0, 2, 1]]  # (X, Z_orig, Y_orig) -> Y-up
    # Los facets del modelo original están en base-1 (convención OBJ);
    # Open3D y el resto de este proyecto esperan índices base-0.
    faces_0indexed = reshaper.facets - 1
    return verts_yup, faces_0indexed