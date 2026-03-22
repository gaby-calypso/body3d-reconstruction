"""
regression_model.py
-------------------
Modelo de regresión lineal multivariable para predicción de medidas
antropométricas a partir de: %grasa corporal, sexo, edad, peso y talla.

Ecuaciones calibradas externamente (ver referencia en docs/).
Todas las medidas predichas están en centímetros.

Inputs:
    body_fat  : float — % de grasa corporal (introducido por el usuario)
    sex       : str   — 'male' (1) o 'female' (0)
    age       : float — años
    weight    : float — kg
    height    : float — metros

Output:
    dict con claves: neck, chest, abdomen, hip, knee, thigh, wrist  (cm)
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class UserInputs:
    """Parámetros introducidos por el usuario para la referencia sintética."""
    body_fat: float   # % grasa corporal
    sex: str          # 'male' o 'female'
    age: float        # años
    weight: float     # kg
    height: float     # metros


def predict_measurements(inputs: UserInputs) -> Dict[str, float]:
    """
    Aplica el modelo de regresión multivariable para predecir las
    7 medidas antropométricas.

    Coeficientes:
        intercept + BodyFat + Sex_numeric + Age + Weight + Height

    Sex_numeric: male=1, female=0

    Returns:
        dict zona → circunferencia predicha en cm
    """
    bf   = inputs.body_fat
    sex  = 1.0 if inputs.sex.lower() == "male" else 0.0
    age  = inputs.age
    w    = inputs.weight
    h    = inputs.height  # en metros (el modelo usa metros)

    predictions = {
        "neck":    31.718385 - 0.041135*bf - 2.712032*sex + 0.023855*age + 0.174929*w  - 4.598225*h,
        "chest":   88.007772 + 0.051081*bf - 4.482891*sex + 0.079188*age + 0.580430*w  - 21.641555*h,
        "abdomen": 73.578446 + 0.345250*bf - 11.288164*sex + 0.103636*age + 0.627832*w - 24.124752*h,
        "hip":     72.286347 + 0.111314*bf + 5.467554*sex  - 0.053214*age + 0.516546*w - 7.860734*h,
        "knee":    18.630646 + 0.028794*bf + 0.751881*sex  + 0.004101*age + 0.143311*w + 4.223780*h,
        "thigh":   56.282174 + 0.038536*bf - 4.308431*sex  - 0.099015*age + 0.370540*w - 13.016735*h,
        "wrist":   13.982496 - 0.030319*bf - 0.607024*sex  + 0.018987*age + 0.074350*w - 1.157423*h,
    }

    return predictions


def print_predictions(predictions: Dict[str, float]) -> None:
    """Imprime las medidas predichas en formato tabla."""
    print("\n  Medidas sintéticas predichas por regresión:")
    print("  " + "─" * 35)
    for zone, cm in predictions.items():
        print(f"  {zone:10s}: {cm:6.1f} cm")
    print("  " + "─" * 35)