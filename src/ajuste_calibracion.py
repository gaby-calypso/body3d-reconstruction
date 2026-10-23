"""
ajuste_calibracion.py
---------------------
Recalcula los factores de src/calibration.py (ZONE_GAIN).

Uso:
    python ajuste_calibracion.py datos.csv

datos.csv (una fila por captura y zona, con encabezado):
    zona,automatica_cm,manual_cm
    cuello,36.2,32.6
    cadera,96.8,87.7
    ...

· `automatica_cm` es la circunferencia SIN compensar: en los resultados del
  pipeline es results[zona]["circumference_raw_cm"].
· Usa solo capturas "normales" (sin cabeza ni distancia alteradas a propósito).

Imprime el bloque ZONE_GAIN para pegar en calibration.py y el error antes y
después de compensar. Factor = Σ(auto·manual) / Σ(auto²) (mínimos cuadrados
sin término constante).
"""
import csv
import sys
from collections import defaultdict

import numpy as np


def main(path):
    datos = defaultdict(lambda: ([], []))
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, m = float(r["automatica_cm"]), float(r["manual_cm"])
            datos[r["zona"].strip()][0].append(a)
            datos[r["zona"].strip()][1].append(m)

    print("ZONE_GAIN = {")
    for zona, (a, m) in datos.items():
        a, m = np.array(a), np.array(m)
        g = float((a * m).sum() / (a * a).sum())
        print(f'    "{zona}": {g:.3f},   # n={len(a)}  '
              f'error medio {np.abs(a - m).mean():.2f} → {np.abs(g * a - m).mean():.2f} cm')
    print("}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])