"""Convierte registros inmutables a matrices de caracteristicas."""
import numpy as np

def separar_caracteristicas_etiquetas(registros):
    """Separar caracteristicas etiquetas.

    Args:
        registros: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    matriz = np.array(registros, dtype=float)
    X = matriz[:, :-1]
    y = matriz[:, -1].astype(int)
    return (X, y)
