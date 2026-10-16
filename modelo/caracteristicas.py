"""Transforma variables numericas y categoricas para KNN."""
import numpy as np
COLUMNAS_NUMERICAS = (1, 4, 7, 10, 12, 15)
COLUMNAS_CATEGORICAS = tuple(filter(lambda indice: indice not in COLUMNAS_NUMERICAS, range(20)))

def obtener_categorias(X_train):
    """Obtener categorias.

    Args:
        X_train: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    return tuple(map(lambda columna: np.unique(X_train[:, columna]), COLUMNAS_CATEGORICAS))

def codificar_one_hot(X, categorias):
    """Codificar one hot.

    Args:
        X: Parametro de entrada.
        categorias: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    columnas_codificadas = tuple(map(lambda elemento: (X[:, elemento[0], np.newaxis] == elemento[1]).astype(float), zip(COLUMNAS_CATEGORICAS, categorias)))
    return np.hstack(columnas_codificadas)

def obtener_limites_numericos(X_train):
    """Obtener limites numericos.

    Args:
        X_train: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    numericas = X_train[:, COLUMNAS_NUMERICAS]
    minimos = np.min(numericas, axis=0)
    maximos = np.max(numericas, axis=0)
    return (minimos, maximos)

def normalizar_numericas(X, minimos, maximos):
    """Normalizar numericas.

    Args:
        X: Parametro de entrada.
        minimos: Parametro de entrada.
        maximos: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    numericas = X[:, COLUMNAS_NUMERICAS]
    diferencias = maximos - minimos
    diferencias_seguras = np.where(diferencias == 0, 1, diferencias)
    return (numericas - minimos) / diferencias_seguras

def unir_caracteristicas(numericas, categoricas):
    """Unir caracteristicas.

    Args:
        numericas: Parametro de entrada.
        categoricas: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    if numericas.shape[0] != categoricas.shape[0]:
        raise ValueError('Las matrices deben tener el mismo numero de registros')
    return np.hstack((numericas, categoricas))
