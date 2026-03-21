"""
smpl_fitting.py
---------------
Genera mallas SMPL con medidas antropométricas específicas.

Estrategia:
    1. Medir el cuerpo promedio SMPL (beta=0)
    2. Optimizar betas normalizados para reproducir medidas objetivo
    3. Permitir transformaciones de zonas individuales o múltiples

Inputs:
    - measurements: dict con medidas del pipeline en cm
    - targets:      dict zona → cm objetivo

Outputs:
    - vertices: (6890, 3) malla 3D en metros
    - faces:    (13776, 3)
    - betas:    (10,)
"""

import numpy as np
import torch
import smplx
from scipy.optimize import minimize


ZONE_Y_METERS = {
    "cuello":  0.42,
    "pecho":   0.22,
    "cintura": 0.05,
    "cadera":  -0.10,
}

TORSO_X_LIMIT = 0.18


def load_smpl_model(models_dir: str = "models",
                    gender: str = "neutral") -> smplx.SMPL:
    """Carga el modelo SMPL."""
    model = smplx.create(
        models_dir,
        model_type="smpl",
        gender=gender,
        num_betas=10,
    )
    model.eval()
    return model


def get_vertices(model: smplx.SMPL,
                  betas: np.ndarray) -> np.ndarray:
    """Genera vértices (6890, 3) en metros."""
    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(betas=betas_t)
    return output.vertices.squeeze(0).numpy()


def measure_zone(vertices: np.ndarray,
                  y_target: float,
                  y_tol: float = 0.025,
                  x_lim: float = TORSO_X_LIMIT) -> float:
    """Mide la circunferencia en una altura Y usando modelo elíptico."""
    mask = (
        (vertices[:, 1] >= y_target - y_tol) &
        (vertices[:, 1] <= y_target + y_tol) &
        (np.abs(vertices[:, 0]) <= x_lim)
    )
    verts = vertices[mask]
    if len(verts) < 4:
        return 0.0
    a = (verts[:, 0].max() - verts[:, 0].min()) / 2
    b = (verts[:, 2].max() - verts[:, 2].min()) / 2
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return float(np.pi * (a + b) * (1 + 3*h / (10 + np.sqrt(4 - 3*h))) * 100)


def get_all_measurements(model: smplx.SMPL,
                          betas: np.ndarray) -> dict:
    """Calcula todas las medidas de la malla con los betas dados."""
    verts = get_vertices(model, betas)
    return {zone: measure_zone(verts, y)
            for zone, y in ZONE_Y_METERS.items()}


def fit_betas_to_targets(model: smplx.SMPL,
                          target_cm: dict,
                          betas_init: np.ndarray | None = None,
                          max_iter: int = 500) -> tuple[np.ndarray, dict]:
    """
    Optimiza betas para alcanzar las medidas objetivo.

    Usa normalización por el cuerpo promedio para estabilidad.

    Args:
        model:       modelo SMPL
        target_cm:   dict zona → cm objetivo
        betas_init:  betas iniciales (None = cuerpo promedio)
        max_iter:    iteraciones máximas

    Returns:
        betas_opt, final_measurements
    """
    if betas_init is None:
        betas_init = np.zeros(10)

    baseline = get_all_measurements(model, np.zeros(10))

    def loss(betas: np.ndarray) -> float:
        meas = get_all_measurements(model, betas)
        err = 0.0
        for zone, target in target_cm.items():
            if zone in meas and meas[zone] > 0 and baseline.get(zone, 0) > 0:
                pred_norm   = meas[zone]   / baseline[zone]
                target_norm = target       / baseline[zone]
                err += (pred_norm - target_norm) ** 2
        err += 0.005 * np.sum(betas ** 2)
        return err

    result = minimize(
        loss,
        betas_init,
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0)] * len(betas_init),
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10}
    )

    betas_opt  = result.x
    final_meas = get_all_measurements(model, betas_opt)
    return betas_opt, final_meas


def generate_smpl_mesh(measurements: dict,
                        models_dir: str = "models",
                        gender: str = "neutral") -> tuple[np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray]:
    """
    Genera malla SMPL que representa a la persona real.

    Args:
        measurements: medidas del pipeline (dict con circumference_cm)
        models_dir:   directorio con modelos
        gender:       'neutral', 'male', 'female'

    Returns:
        vertices, faces, betas
    """
    model = load_smpl_model(models_dir, gender)
    target = {z: d["circumference_cm"] for z, d in measurements.items()}

    print("  Cuerpo promedio SMPL:")
    baseline = get_all_measurements(model, np.zeros(10))
    for z, v in baseline.items():
        print(f"    {z:10s}: {v:.1f} cm")

    print("\n  Objetivo (pipeline):")
    for z, v in target.items():
        print(f"    {z:10s}: {v:.1f} cm")

    betas_opt, final_meas = fit_betas_to_targets(model, target)

    print("\n  Resultados:")
    print("  " + "-" * 48)
    print(f"  {'Zona':10s} {'Objetivo':>10s} {'SMPL':>10s} {'Error':>8s}")
    print("  " + "-" * 48)
    for zone in target:
        obj  = target[zone]
        pred = final_meas.get(zone, 0)
        err  = pred - obj
        sign = "+" if err >= 0 else ""
        print(f"  {zone:10s} {obj:>8.1f}cm {pred:>8.1f}cm "
              f"{sign}{err:>6.1f}cm")
    print("  " + "-" * 48)

    return get_vertices(model, betas_opt), model.faces, betas_opt


def transform_smpl_mesh(betas_base: np.ndarray,
                         measurements_base: dict,
                         targets: dict,
                         models_dir: str = "models",
                         gender: str = "neutral") -> tuple[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray]:
    """
    Transforma una malla SMPL existente a nuevas medidas objetivo.

    Las zonas no especificadas en targets mantienen sus medidas.

    Args:
        betas_base:        betas de la persona real
        measurements_base: medidas actuales de la persona
        targets:           dict zona → cm objetivo
        models_dir:        directorio con modelos
        gender:            género del modelo

    Returns:
        vertices, faces, betas_new
    """
    model = load_smpl_model(models_dir, gender)

    full_targets = {
        z: d["circumference_cm"] for z, d in measurements_base.items()
    }
    full_targets.update(targets)

    print(f"  Zonas a modificar: {list(targets.keys())}")
    for zone, target_cm in targets.items():
        current = measurements_base.get(zone, {}).get("circumference_cm", 0)
        print(f"    {zone:10s}: {current:.1f}cm → {target_cm:.1f}cm")

    betas_new, final_meas = fit_betas_to_targets(
        model, full_targets, betas_init=betas_base
    )

    print("\n  Medidas finales:")
    print("  " + "-" * 35)
    for zone in full_targets:
        print(f"  {zone:10s}: {final_meas.get(zone, 0):.1f} cm")
    print("  " + "-" * 35)

    return get_vertices(model, betas_new), model.faces, betas_new