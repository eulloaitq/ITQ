"""Calcula distancias euclidianas mediante algebra matricial."""
import numpy as np

def calcular_distancias_euclidianas(X_test, X_train):
    """Calcular distancias euclidianas.

    Parameters
    ----------
    X_test : numpy.ndarray
        Matriz de prueba (n_prueba, n_atributos).
    X_train : numpy.ndarray
        Matriz de entrenamiento (n_train, n_atributos).

    Returns
    -------
    numpy.ndarray
        Distancias (n_prueba, n_train).
"""
    cuadrados_test = np.sum(X_test ** 2, axis=1, keepdims=True)
    cuadrados_train = np.sum(X_train ** 2, axis=1)
    producto = X_test @ X_train.T
    distancias_cuadradas = cuadrados_test + cuadrados_train - 2 * producto
    distancias_cuadradas = np.maximum(distancias_cuadradas, 0)
    distancias = np.sqrt(distancias_cuadradas)
    return distancias
