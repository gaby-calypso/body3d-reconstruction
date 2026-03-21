"""
morphing.py
-----------
Aplica transformaciones paramétricas a la nube de puntos 3D
para generar una malla de referencia con medidas objetivo.

Estrategia:
    Para cada zona (cintura, cadera, pecho, cuello):
    1. Calcular el factor de escala = contorno_objetivo / contorno_actual
    2. Expandir/contraer los puntos en X y Z (no en Y — la altura no cambia)
    3. Usar una función gaussiana para suavizar la transición entre zonas

    Varias zonas pueden transformarse simultáneamente — sus influencias
    gaussianas se combinan por suma ponderada.

Inputs:
    - points:       np.ndarray (N, 3) float32 — nube de puntos original
    - measurements: dict con medidas actuales por zona
    - targets:      dict con contornos objetivo en cm por zona

Outputs:
    - points_morphed: np.ndarray (N, 3) float32 — nube transformada
"""

import numpy as np


# ── Coordenadas Y directas en el espacio 3D (mm) — medidas empíricamente ─────
# Derivadas de: y_world = (y_img - cy) * depth_ref / fy
ZONE_Y_WORLD = {
    "cuello":  -169.0,
    "pecho":     35.0,
    "cintura":  203.0,
    "cadera":   293.0,
}

# Radio de influencia gaussiana en mm del mundo 3D
# Zona entre zonas adyacentes / 3 para transición suave pero localizada
ZONE_SIGMA = {
    "cuello":   60.0,
    "pecho":    80.0,
    "cintura":  70.0,
    "cadera":   70.0,
}

# Parámetros intrínsecos D455 para convertir Y imagen → Y mundo
FY = 649.46
CY = 360.00

def gaussian_influence(y_world: np.ndarray,
                        y_center: float,
                        sigma: float) -> np.ndarray:
    """
    Calcula la influencia gaussiana de una zona sobre cada punto.

    Cada punto recibe un peso entre 0 y 1 según su distancia
    al centro de la zona. Puntos lejos de la zona → peso ~0.
    Puntos en el centro de la zona → peso ~1.

    Args:
        y_world:  coordenadas Y de todos los puntos en mm
        y_center: coordenada Y del centro de la zona en mm
        sigma:    radio de influencia en mm

    Returns:
        np.ndarray (N,) con pesos entre 0 y 1
    """
    return np.exp(-((y_world - y_center) ** 2) / (2 * sigma ** 2))


def compute_scale_factor(current_cm: float, target_cm: float) -> float:
    """
    Calcula el factor de escala para alcanzar el contorno objetivo.

    El factor se aplica a los ejes X y Z (ancho y profundidad).
    Para una elipse, escalar X y Z por un factor k multiplica
    el perímetro aproximadamente por k.

    Args:
        current_cm: contorno actual en cm
        target_cm:  contorno objetivo en cm

    Returns:
        factor de escala (>1 expande, <1 contrae)
    """
    if current_cm <= 0:
        raise ValueError("El contorno actual debe ser mayor que 0.")
    return target_cm / current_cm


def apply_morphing(points: np.ndarray,
                   measurements: dict,
                   targets: dict,
                   ref_depth_mm: float = 1500.0) -> np.ndarray:
    """
    Función principal: aplica las transformaciones a la nube de puntos.

    Para cada zona con un objetivo definido:
    1. Calcula el factor de escala
    2. Calcula la influencia gaussiana sobre todos los puntos
    3. Combina las influencias de todas las zonas
    4. Aplica la escala ponderada en X y Z

    Args:
        points:       np.ndarray (N, 3) float32 — X, Y, Z en mm
        measurements: dict con medidas actuales (de extract_measurements)
        targets:      dict zona → contorno objetivo en cm
                      ejemplo: {"cintura": 50.0, "cadera": 95.0}
        ref_depth_mm: profundidad de referencia para convertir Y

    Returns:
        points_morphed: np.ndarray (N, 3) float32 — nube transformada

    Raises:
        ValueError: si una zona objetivo no existe en measurements
    """
    points_morphed = points.copy().astype(np.float64)
    Y_world = points[:, 1]  # coordenada Y de cada punto en mm

    # Acumuladores para la transformación ponderada
    scale_accumulated = np.zeros(len(points), dtype=np.float64)
    weight_total      = np.zeros(len(points), dtype=np.float64)

    for zone_name, target_cm in targets.items():

        if zone_name not in measurements:
            raise ValueError(
                f"Zona '{zone_name}' no encontrada en measurements. "
                f"Zonas disponibles: {list(measurements.keys())}"
            )

        current_cm = measurements[zone_name]["circumference_cm"]
        scale      = compute_scale_factor(current_cm, target_cm)

        print(f"  {zone_name:8s}: {current_cm:.1f}cm → {target_cm:.1f}cm "
              f"(factor={scale:.3f})")

        # Centro de la zona directamente en mm del mundo 3D
        y_center = ZONE_Y_WORLD[zone_name]
        sigma_mm = ZONE_SIGMA[zone_name]

        # Influencia gaussiana de esta zona
        influence = gaussian_influence(Y_world, y_center, sigma_mm)

        # Acumular escala ponderada
        scale_accumulated += influence * scale
        weight_total      += influence

    # Evitar división por cero
    safe_weights = np.where(weight_total > 1e-6, weight_total, 1.0)
    scale_per_point = scale_accumulated / safe_weights

    # Donde no hay influencia de ninguna zona → escala = 1 (sin cambio)
    scale_per_point = np.where(weight_total > 1e-6, scale_per_point, 1.0)

    # Aplicar escala en X y Z solamente (Y no cambia — altura constante)
    points_morphed[:, 0] *= scale_per_point  # X — ancho
    points_morphed[:, 2] *= scale_per_point  # Z — profundidad

    return points_morphed.astype(np.float32)


def morph_pointcloud(points: np.ndarray,
                      measurements: dict,
                      targets: dict) -> tuple[np.ndarray, dict]:
    """
    Wrapper de alto nivel: aplica morfing y calcula las nuevas medidas.

    Args:
        points:       np.ndarray (N, 3) float32
        measurements: medidas actuales
        targets:      dict zona → contorno objetivo en cm

    Returns:
        points_morphed: nube transformada (N, 3) float32
        new_measurements: dict con las medidas después del morfing
    """
    print(f"  Zonas a transformar: {list(targets.keys())}")

    points_morphed = apply_morphing(points, measurements, targets)

    # Estimar nuevas medidas (aproximación geométrica)
    new_measurements = {}
    for zone_name, data in measurements.items():
        if zone_name in targets:
            # Zona transformada — usamos el objetivo directamente
            new_cm = targets[zone_name]
        else:
            # Zona no transformada — puede haber cambio leve por suavizado
            new_cm = data["circumference_cm"]

        new_measurements[zone_name] = {
            **data,
            "circumference_cm": round(new_cm, 1),
            "circumference_mm": round(new_cm * 10, 1),
        }

    return points_morphed, new_measurements
