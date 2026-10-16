"""Limpieza funcional, imputacion y deteccion de atipicos."""

def es_valor_ausente(valor):
    """Es valor ausente.

    Args:
        valor: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    return valor is None or valor == ''

def contar_ausentes(registros):
    """Contar ausentes.

    Args:
        registros: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    return sum(map(lambda registro: sum(map(es_valor_ausente, registro)), registros))

def simular_ausentes(registros):
    """Simular ausentes.

    Args:
        registros: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    return tuple(map(lambda elemento: tuple(map(lambda posicion: None if elemento[0] % 20 == 0 and posicion == 4 else elemento[1][posicion], range(len(elemento[1])))), enumerate(registros)))

def calcular_mediana(valores):
    """Calcular mediana.

    Args:
        valores: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    validos = tuple(sorted(filter(lambda valor: not es_valor_ausente(valor), valores)))
    cantidad = len(validos)
    if cantidad == 0:
        raise ValueError('No existen valores para calcular la mediana')
    mitad = cantidad // 2
    if cantidad % 2 == 0:
        return (validos[mitad - 1] + validos[mitad]) / 2
    return validos[mitad]

def imputar_columna(registros, columna, mediana):
    """Imputar columna.

    Args:
        registros: Parametro de entrada.
        columna: Parametro de entrada.
        mediana: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    return tuple(map(lambda registro: tuple(map(lambda posicion: mediana if posicion == columna and es_valor_ausente(registro[posicion]) else registro[posicion], range(len(registro)))), registros))

def calcular_iqr(registros, columna):
    """Calcular iqr.

    Args:
        registros: Parametro de entrada.
        columna: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    valores = tuple(sorted(map(lambda registro: registro[columna], registros)))
    cantidad = len(valores)
    if cantidad < 4:
        raise ValueError('Se requieren al menos 4 valores')

    def percentil(p):
        """Percentil.

    Args:
        p: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
        posicion = (cantidad - 1) * p
        inferior = int(posicion)
        superior = min(inferior + 1, cantidad - 1)
        fraccion = posicion - inferior
        return valores[inferior] * (1 - fraccion) + valores[superior] * fraccion
    q1 = percentil(0.25)
    q3 = percentil(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return (q1, q3, iqr, limite_inferior, limite_superior)

def filtrar_outliers(registros, columna, limite_inferior, limite_superior):
    """Filtrar outliers.

    Args:
        registros: Parametro de entrada.
        columna: Parametro de entrada.
        limite_inferior: Parametro de entrada.
        limite_superior: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    return tuple(filter(lambda registro: limite_inferior <= registro[columna] <= limite_superior, registros))
