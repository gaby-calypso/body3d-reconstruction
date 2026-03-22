"""
smpl_fitting.py
---------------
Genera mallas SMPL con medidas antropométricas específicas.

Estrategia de medición:
    Modelo elíptico de Ramanujan sobre los percentiles 5-95 de X y Z
    en una banda Y estrecha. Esto es robusto a vértices dispersos y
    produce circunferencias consistentes con medidas anatómicas reales.

Calibración (beta=0, T-pose, género neutral):
    cuello  → Y=0.395, ancho_X=15.9cm, prof_Z=21.0cm → circ~59cm convexhull
              pero el cuello anatómico real usa solo los vértices del cilindro
              central → x_lim=0.065, z usando percentil → ~35cm
"""

import numpy as np
import torch
import smplx
from scipy.optimize import minimize, differential_evolution


# ── Configuración de zonas calibrada con datos reales DEBUG ──────────────────
# y        : altura Y del corte (metros, beta=0)
# y_tol    : semiancho de la banda Y
# x_lim    : límite en |X| para excluir brazos/piernas
# pct      : percentil para semiejes (5-95 → robusto a outliers)
ZONE_CONFIG = {
    "cuello":  {"y":  0.395, "y_tol": 0.018, "x_lim": 0.065, "pct": 5},
    "pecho":   {"y":  0.175, "y_tol": 0.025, "x_lim": 0.175, "pct": 5},
    "cintura": {"y":  0.030, "y_tol": 0.025, "x_lim": 0.155, "pct": 5},
    "cadera":  {"y": -0.095, "y_tol": 0.025, "x_lim": 0.155, "pct": 5},
}


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
                  y_tol: float,
                  x_lim: float,
                  pct: float = 5) -> float:
    """
    Mide la circunferencia de una zona corporal usando el modelo
    elíptico de Ramanujan con percentiles robustos.

    Los semiejes a (X) y b (Z) se calculan con percentiles pct y 100-pct
    para excluir outliers y vértices dispersos que inflan la medida.

    Args:
        vertices : (N, 3) vértices en metros
        y_target : altura Y del corte
        y_tol    : semiancho de la banda Y
        x_lim    : límite |X| para excluir extremidades
        pct      : percentil inferior (pct=5 → usa p5 y p95)

    Returns:
        circunferencia en cm
    """
    mask = (
        (vertices[:, 1] >= y_target - y_tol) &
        (vertices[:, 1] <= y_target + y_tol) &
        (np.abs(vertices[:, 0]) <= x_lim)
    )
    pts = vertices[mask]

    if len(pts) < 6:
        return 0.0

    # Semiejes con percentiles para robustez a outliers
    x_lo, x_hi = np.percentile(pts[:, 0], [pct, 100 - pct])
    z_lo, z_hi = np.percentile(pts[:, 2], [pct, 100 - pct])

    a = (x_hi - x_lo) / 2.0   # semieje X
    b = (z_hi - z_lo) / 2.0   # semieje Z

    if a <= 0 or b <= 0:
        return 0.0

    # Fórmula de Ramanujan para perímetro de elipse
    h = ((a - b) / (a + b)) ** 2
    circ_m = np.pi * (a + b) * (1 + 3*h / (10 + np.sqrt(4 - 3*h)))

    return float(circ_m * 100)   # metros → cm


def get_all_measurements(model: smplx.SMPL,
                          betas: np.ndarray) -> dict:
    """Calcula todas las medidas de la malla con los betas dados."""
    verts = get_vertices(model, betas)
    return {
        zone: measure_zone(
            verts,
            cfg["y"], cfg["y_tol"], cfg["x_lim"], cfg["pct"]
        )
        for zone, cfg in ZONE_CONFIG.items()
    }


def _calibrate_baseline() -> dict:
    """
    Valores de referencia medidos con beta=0, género neutral.
    Pre-calculados para evitar recalcular en cada optimización.
    Actualizar si se cambia ZONE_CONFIG.
    """
    return {
        "cuello":  30.2,
        "pecho":   82.4,
        "cintura": 77.1,
        "cadera":  76.8,
    }


def fit_betas_to_targets(model: smplx.SMPL,
                          target_cm: dict,
                          betas_init: np.ndarray | None = None,
                          max_iter: int = 2000) -> tuple[np.ndarray, dict]:
    """
    Optimiza betas para alcanzar las medidas objetivo.

    Estrategia en dos fases:
        Fase 1 — Evolución diferencial global (escapa mínimos locales)
        Fase 2 — L-BFGS-B local desde el mejor punto de fase 1

    Args:
        model      : modelo SMPL cargado
        target_cm  : dict zona → cm objetivo
        betas_init : betas iniciales (None → zeros)
        max_iter   : iteraciones máximas fase local

    Returns:
        betas_opt, final_measurements
    """
    if betas_init is None:
        betas_init = np.zeros(10)

    # Pesos por zona — mayor peso a zonas con más varianza en SMPL
    zone_weights = {
        "cuello":  4.0,
        "pecho":   1.5,
        "cintura": 2.0,
        "cadera":  2.0,
    }

    def loss(betas: np.ndarray) -> float:
        meas = get_all_measurements(model, betas)
        err  = 0.0
        for zone, target in target_cm.items():
            if zone in meas and meas[zone] > 0:
                w    = zone_weights.get(zone, 1.0)
                diff = (meas[zone] - target) / 10.0
                err += w * diff ** 2
        err += 0.0005 * np.sum(betas ** 2)
        return err

    bounds = [(-10.0, 10.0)] * len(betas_init)

    # Fase 1: búsqueda global con evolución diferencial
    print("    Optimizando forma corporal (fase 1 — búsqueda global)...")
    result_global = differential_evolution(
        loss,
        bounds,
        maxiter=300,
        popsize=8,
        tol=1e-8,
        seed=42,
        workers=1,
        polish=False,
    )

    # Fase 2: refinamiento local desde el mejor punto global
    print("    Refinando (fase 2 — optimización local)...")
    result_local = minimize(
        loss,
        result_global.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-13}
    )

    betas_opt  = result_local.x
    final_meas = get_all_measurements(model, betas_opt)
    return betas_opt, final_meas


def generate_smpl_mesh(measurements: dict,
                        models_dir: str = "models",
                        gender: str = "neutral") -> tuple[np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray]:
    """
    Genera malla SMPL ajustada a las medidas dadas.

    Args:
        measurements : dict zona → {"circumference_cm": float}
        models_dir   : directorio con modelos SMPL
        gender       : 'neutral', 'male', 'female'

    Returns:
        vertices (6890,3), faces (13776,3), betas (10,)
    """
    model  = load_smpl_model(models_dir, gender)
    target = {z: d["circumference_cm"] for z, d in measurements.items()}

    print("  Cuerpo promedio SMPL:")
    baseline = get_all_measurements(model, np.zeros(10))
    for z, v in baseline.items():
        print(f"    {z:10s}: {v:.1f} cm")

    print("\n  Objetivo:")
    for z, v in target.items():
        print(f"    {z:10s}: {v:.1f} cm")

    # Intentar usar cache para evitar reoptimizar
    from src.smpl_cache import load_cached_betas, save_cached_betas
    cached = load_cached_betas(target)
    if cached is not None:
        print("  ✓ Usando betas cacheados (sin reoptimizar)")
        betas_opt  = cached
        final_meas = get_all_measurements(model, betas_opt)
    else:
        betas_opt, final_meas = fit_betas_to_targets(model, target)
        save_cached_betas(target, betas_opt)
        print("  ✓ Betas guardados en cache")
        
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

    Args:
        betas_base        : betas de la persona base
        measurements_base : dict zona → {"circumference_cm": float}
        targets           : dict zona → cm objetivo (zonas a modificar)
        models_dir        : directorio con modelos
        gender            : género del modelo

    Returns:
        vertices, faces, betas_new
    """
    model = load_smpl_model(models_dir, gender)

    full_targets = {
        z: d["circumference_cm"] for z, d in measurements_base.items()
    }
    full_targets.update(targets)

    print(f"  Zonas a modificar: {list(targets.keys())}")
    for zone, target_cm_val in targets.items():
        current = measurements_base.get(zone, {}).get("circumference_cm", 0)
        print(f"    {zone:10s}: {current:.1f}cm → {target_cm_val:.1f}cm")

    betas_new, final_meas = fit_betas_to_targets(
        model, full_targets, betas_init=betas_base
    )

    print("\n  Medidas finales:")
    print("  " + "-" * 35)
    for zone in full_targets:
        print(f"  {zone:10s}: {final_meas.get(zone, 0):.1f} cm")
    print("  " + "-" * 35)

    return get_vertices(model, betas_new), model.faces, betas_new


def debug_zone(model: smplx.SMPL, zone: str = "cuello") -> None:
    """Diagnóstico: imprime vértices capturados y medida resultante."""
    verts = get_vertices(model, np.zeros(10))
    cfg   = ZONE_CONFIG[zone]
    mask  = (
        (verts[:, 1] >= cfg["y"] - cfg["y_tol"]) &
        (verts[:, 1] <= cfg["y"] + cfg["y_tol"]) &
        (np.abs(verts[:, 0]) <= cfg["x_lim"])
    )
    pts = verts[mask]
    circ = measure_zone(verts, cfg["y"], cfg["y_tol"], cfg["x_lim"], cfg["pct"])
    print(f"\n  [DEBUG] zona={zone}  n_verts={len(pts)}  circ={circ:.1f}cm")
    if len(pts) > 0:
        print(f"    X p5-p95: [{np.percentile(pts[:,0],5)*100:.1f}, "
              f"{np.percentile(pts[:,0],95)*100:.1f}] cm")
        print(f"    Z p5-p95: [{np.percentile(pts[:,2],5)*100:.1f}, "
              f"{np.percentile(pts[:,2],95)*100:.1f}] cm")