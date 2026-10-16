"""Divide registros de forma reproducible sin modificar los originales."""
import random

def dividir_datos(registros, proporcion=0.8, semilla=42):
    """Dividir datos.

    Args:
        registros: Parametro de entrada.
        proporcion: Parametro de entrada.
        semilla: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    if not 0 < proporcion < 1:
        raise ValueError('La proporcion debe estar entre 0 y 1')
    generador = random.Random(semilla)
    indices = list(range(len(registros)))
    generador.shuffle(indices)
    limite = int(len(indices) * proporcion)
    entrenamiento = tuple(map(lambda i: registros[i], indices[:limite]))
    prueba = tuple(map(lambda i: registros[i], indices[limite:]))
    return (entrenamiento, prueba)
