"""
smpl_cache.py
-------------
Cache de betas SMPL para evitar reoptimizar cuando los parámetros
no han cambiado. Persiste en output/.smpl_cache.npz entre sesiones.
"""

import numpy as np
import hashlib
import json
from pathlib import Path

CACHE_PATH = Path("output/.smpl_cache.npz")


def _params_hash(target_cm: dict) -> str:
    """Hash MD5 — redondea a 1 decimal para tolerancia numérica."""
    rounded = {k: round(float(v), 1) for k, v in sorted(target_cm.items())}
    key = json.dumps(rounded, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def load_cached_betas(target_cm: dict) -> np.ndarray | None:
    """Devuelve betas cacheados si el hash coincide, None si no."""
    if not CACHE_PATH.exists():
        return None
    try:
        data      = np.load(str(CACHE_PATH), allow_pickle=True)
        cached_h  = str(data["hash"])
        current_h = _params_hash(target_cm)
        if cached_h == current_h:
            print("  ✓ Cache SMPL válido — saltando optimización (~0s)")
            return data["betas"]
        else:
            print(f"  Cache inválido — parámetros cambiaron")
            print(f"    guardado : {cached_h}")
            print(f"    actual   : {current_h}")
    except Exception as e:
        print(f"  Error leyendo cache: {e}")
    return None


def save_cached_betas(target_cm: dict, betas: np.ndarray) -> None:
    """Guarda betas y hash en cache."""
    CACHE_PATH.parent.mkdir(exist_ok=True)
    h = _params_hash(target_cm)
    np.savez(str(CACHE_PATH), betas=betas, hash=np.array(h))
    print(f"  ✓ Cache SMPL guardado")
    