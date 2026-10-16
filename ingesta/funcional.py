"""Compone transformaciones funcionales puras."""
from functools import reduce

def pipe(valor, *funciones):
    """Pipe.

    Args:
        valor: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    return reduce(lambda acumulado, funcion: funcion(acumulado), funciones, valor)
