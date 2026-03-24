"""
smpl_cache.py
-------------
Cache de betas SMPL para evitar reoptimizar cuando los parámetros
no han cambiado. Persiste en output/.smpl_cache.json entre sesiones.

Soporta múltiples entradas — cada combinación de medidas + género
tiene su propio hash. El género forma parte del hash para que
cambiar de neutral a male/female invalide el cache correctamente.
"""

import numpy as np
import hashlib
import json
from pathlib import Path

CACHE_PATH = Path("output/.smpl_cache.json")


def _params_hash(target_cm: dict, gender: str = "neutral") -> str:
    """
    Hash MD5 que incluye medidas redondeadas y género.
    Redondea a 1 decimal para tolerancia numérica.
    """
    rounded = {k: round(float(v), 1) for k, v in sorted(target_cm.items())}
    rounded["__gender__"] = gender.lower()
    key = json.dumps(rounded, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def _load_cache() -> dict:
    """Carga el cache completo desde disco. Devuelve dict vacío si no existe."""
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  ⚠ Error leyendo cache SMPL: {e}")
        return {}


def _save_cache(data: dict) -> None:
    """Guarda el cache completo a disco."""
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)


def load_cached_betas(target_cm: dict,
                      gender: str = "neutral") -> np.ndarray | None:
    """
    Devuelve betas cacheados si el hash coincide, None si no.

    Args:
        target_cm: dict zona → cm objetivo
        gender:    género del modelo ('neutral', 'male', 'female')

    Returns:
        np.ndarray de betas si está en cache, None si no
    """
    cache = _load_cache()
    h     = _params_hash(target_cm, gender)

    if h in cache:
        betas = np.array(cache[h])
        print(f"  ✓ Cache SMPL válido [{gender}] — saltando optimización")
        return betas

    return None


def save_cached_betas(target_cm: dict,
                      betas: np.ndarray,
                      gender: str = "neutral") -> None:
    """
    Guarda betas en el cache. No sobreescribe otras entradas.

    Args:
        target_cm: dict zona → cm objetivo
        betas:     np.ndarray de betas optimizados
        gender:    género del modelo
    """
    cache = _load_cache()
    h     = _params_hash(target_cm, gender)
    cache[h] = betas.tolist()
    _save_cache(cache)
    print(f"  ✓ Cache SMPL guardado [{gender}] ({len(cache)} entradas)")


def clear_cache() -> None:
    """Borra todo el cache. Útil para forzar reoptimización."""
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print("  ✓ Cache SMPL borrado")
    else:
        print("  Cache ya estaba vacío")