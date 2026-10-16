"""Calcula metricas de equidad por grupos de edad."""
import numpy as np

def calcular_disparate_impact(edades, predicciones):
    """Calcular disparate impact.

    Args:
        edades: Parametro de entrada.
        predicciones: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    if len(edades) != len(predicciones):
        raise ValueError('Edades y predicciones deben tener la misma longitud')
    grupo_a = edades < 25
    grupo_b = edades >= 25
    total_a = np.sum(grupo_a)
    total_b = np.sum(grupo_b)
    favorables_a = np.sum(grupo_a & (predicciones == 1))
    favorables_b = np.sum(grupo_b & (predicciones == 1))
    tasa_a = favorables_a / total_a if total_a > 0 else np.nan
    tasa_b = favorables_b / total_b if total_b > 0 else np.nan
    disparate_impact = tasa_a / tasa_b if tasa_b > 0 else np.nan
    return {'total_a': int(total_a), 'total_b': int(total_b), 'favorables_a': int(favorables_a), 'favorables_b': int(favorables_b), 'tasa_a': tasa_a, 'tasa_b': tasa_b, 'disparate_impact': disparate_impact}

def calcular_exactitud_por_grupo(edades, y_real, y_pred):
    """Calcular exactitud por grupo.

    Args:
        edades: Parametro de entrada.
        y_real: Parametro de entrada.
        y_pred: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    if not len(edades) == len(y_real) == len(y_pred):
        raise ValueError('Los vectores deben tener la misma longitud')
    grupo_a = edades < 25
    grupo_b = edades >= 25
    correctas = y_real == y_pred
    total_a = np.sum(grupo_a)
    total_b = np.sum(grupo_b)
    correctas_a = np.sum(grupo_a & correctas)
    correctas_b = np.sum(grupo_b & correctas)
    exactitud_a = correctas_a / total_a if total_a > 0 else np.nan
    exactitud_b = correctas_b / total_b if total_b > 0 else np.nan
    return {'total_a': int(total_a), 'total_b': int(total_b), 'correctas_a': int(correctas_a), 'correctas_b': int(correctas_b), 'exactitud_a': exactitud_a, 'exactitud_b': exactitud_b}

def calcular_falsos_negativos_por_grupo(edades, y_real, y_pred):
    """Calcular falsos negativos por grupo.

    Args:
        edades: Parametro de entrada.
        y_real: Parametro de entrada.
        y_pred: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    if not len(edades) == len(y_real) == len(y_pred):
        raise ValueError('Los vectores deben tener la misma longitud')
    grupo_a = edades < 25
    grupo_b = edades >= 25
    malos_creditos = y_real == 0
    falsos_negativos = (y_real == 0) & (y_pred == 1)
    malos_a = np.sum(grupo_a & malos_creditos)
    malos_b = np.sum(grupo_b & malos_creditos)
    fn_a = np.sum(grupo_a & falsos_negativos)
    fn_b = np.sum(grupo_b & falsos_negativos)
    tasa_fn_a = fn_a / malos_a if malos_a > 0 else np.nan
    tasa_fn_b = fn_b / malos_b if malos_b > 0 else np.nan
    return {'malos_a': int(malos_a), 'malos_b': int(malos_b), 'fn_a': int(fn_a), 'fn_b': int(fn_b), 'tasa_fn_a': tasa_fn_a, 'tasa_fn_b': tasa_fn_b}
