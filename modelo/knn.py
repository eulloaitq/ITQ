"""Predice clases mediante los vecinos mas cercanos."""
import numpy as np
from modelo.distancias import calcular_distancias_euclidianas

def obtener_vecinos_cercanos(distancias, k=5):
    """Obtener vecinos cercanos.

    Args:
        distancias: Parametro de entrada.
        k: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    if not 1 <= k <= distancias.shape[1]:
        raise ValueError('K debe estar entre 1 y el numero de registros de entrenamiento')
    indices_ordenados = np.argsort(distancias, axis=1)
    vecinos = indices_ordenados[:, :k]
    return vecinos

def votacion_mayoritaria(indices_vecinos, y_train):
    """Votacion mayoritaria.

    Args:
        indices_vecinos: Parametro de entrada.
        y_train: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    etiquetas_vecinos = y_train[indices_vecinos]
    votos_clase_0 = np.sum(etiquetas_vecinos == 0, axis=1)
    votos_clase_1 = np.sum(etiquetas_vecinos == 1, axis=1)
    predicciones = np.where(votos_clase_1 > votos_clase_0, 1, 0)
    return predicciones

def predecir_knn(X_train, y_train, X_test, k=5):
    """Predecir knn.

    Args:
        X_train: Parametro de entrada.
        y_train: Parametro de entrada.
        X_test: Parametro de entrada.
        k: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    distancias = calcular_distancias_euclidianas(X_test, X_train)
    indices_vecinos = obtener_vecinos_cercanos(distancias, k)
    predicciones = votacion_mayoritaria(indices_vecinos, y_train)
    return predicciones
