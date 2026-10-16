"""Normalizacion y estadisticas de columnas."""

def calcular_min_max(registros, columna):
    """Calcular min max.

    Args:
        registros: Parametro de entrada.
        columna: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    valores = tuple(map(lambda registro: registro[columna], registros))
    return (min(valores), max(valores))

def normalizar_min_max(registros, columna, minimo, maximo):
    """Normalizar min max.

    Args:
        registros: Parametro de entrada.
        columna: Parametro de entrada.
        minimo: Parametro de entrada.
        maximo: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    diferencia = maximo - minimo
    return tuple(map(lambda registro: tuple(map(lambda posicion: (registro[posicion] - minimo) / diferencia if posicion == columna and diferencia != 0 else 0.0 if posicion == columna else registro[posicion], range(len(registro)))), registros))
