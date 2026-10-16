"""Preprocesamiento de entrenamiento y prueba sin fuga de datos."""
from ingesta.limpieza import calcular_mediana, imputar_columna, calcular_iqr, filtrar_outliers
from ingesta.transformaciones import calcular_min_max, normalizar_min_max
from ingesta.funcional import pipe

def preparar_datos(entrenamiento, prueba, columna=4):
    """Preparar datos.

    Args:
        entrenamiento: Parametro de entrada.
        prueba: Parametro de entrada.
        columna: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    valores = tuple(map(lambda registro: registro[columna], entrenamiento))
    mediana = calcular_mediana(valores)
    entrenamiento_imputado = imputar_columna(entrenamiento, columna, mediana)
    q1, q3, iqr, minimo_iqr, maximo_iqr = calcular_iqr(entrenamiento_imputado, columna)
    entrenamiento_filtrado = filtrar_outliers(entrenamiento_imputado, columna, minimo_iqr, maximo_iqr)
    minimo, maximo = calcular_min_max(entrenamiento_filtrado, columna)
    entrenamiento_final = pipe(entrenamiento, lambda datos: imputar_columna(datos, columna, mediana), lambda datos: filtrar_outliers(datos, columna, minimo_iqr, maximo_iqr), lambda datos: normalizar_min_max(datos, columna, minimo, maximo))
    prueba_final = pipe(prueba, lambda datos: imputar_columna(datos, columna, mediana), lambda datos: normalizar_min_max(datos, columna, minimo, maximo))
    return (entrenamiento_final, prueba_final)
