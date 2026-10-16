"""Clasifica nuevas solicitudes con el KNN entrenado en esta ejecucion.

No se entrena con las solicitudes nuevas. Los parametros se obtienen
exclusivamente del conjunto de entrenamiento reproducible (semilla 42).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ingesta.division import dividir_datos
from ingesta.limpieza import calcular_iqr, calcular_mediana, filtrar_outliers, imputar_columna, simular_ausentes
from ingesta.transformaciones import calcular_min_max, normalizar_min_max
from modelo.caracteristicas import (codificar_one_hot, normalizar_numericas,
                                    obtener_categorias, obtener_limites_numericos,
                                    unir_caracteristicas)
from modelo.datos import separar_caracteristicas_etiquetas
from modelo.knn import predecir_knn

logger = logging.getLogger(__name__)
COLUMNAS = ("laufkont", "laufzeit", "moral", "verw", "hoehe", "sparkont",
            "beszeit", "rate", "famges", "buerge", "wohnzeit", "verm",
            "alter", "weitkred", "wohn", "bishkred", "beruf", "pers",
            "telef", "gastarb")


def entrenar_modelo():
    """Reconstruye el modelo original sin usar datos de solicitudes nuevas.

    Returns:
        dict: Matriz de entrenamiento, etiquetas y parametros aprendidos.
    """
    ruta = Path(__file__).resolve().parent / "datos" / "SouthGermanCredit.asc"
    with ruta.open(encoding="utf-8-sig") as archivo:
        lineas = archivo.readlines()
    registros = tuple(tuple(map(int, linea.split())) for linea in lineas[1:])
    simulados = simular_ausentes(registros)
    entrenamiento, _ = dividir_datos(simulados, proporcion=0.8, semilla=42)
    mediana = calcular_mediana(tuple(fila[4] for fila in entrenamiento))
    imputados = imputar_columna(entrenamiento, 4, mediana)
    _, _, _, inferior, superior = calcular_iqr(imputados, 4)
    filtrados = filtrar_outliers(imputados, 4, inferior, superior)
    minimo, maximo = calcular_min_max(filtrados, 4)
    normalizados = normalizar_min_max(filtrados, 4, minimo, maximo)
    x_train, y_train = separar_caracteristicas_etiquetas(normalizados)
    categorias = obtener_categorias(x_train)
    minimos, maximos = obtener_limites_numericos(x_train)
    x_final = unir_caracteristicas(
        normalizar_numericas(x_train, minimos, maximos),
        codificar_one_hot(x_train, categorias),
    )
    logger.info("Modelo listo: %s registros y %s caracteristicas", *x_final.shape)
    return {"x": x_final, "y": y_train, "mediana": mediana,
            "minimo": minimo, "maximo": maximo, "categorias": categorias,
            "minimos": minimos, "maximos": maximos}


def clasificar_solicitud(modelo, solicitud):
    """Clasifica una solicitud sin recalcular parametros del entrenamiento.

    Args:
        modelo (dict): Modelo y transformaciones aprendidas.
        solicitud (dict): Valores de las 20 caracteristicas originales.

    Returns:
        int: Clase 0 (mal credito) o 1 (buen credito).

    Raises:
        ValueError: Si faltan campos o los valores no son validos.
    """
    if set(solicitud) != set(COLUMNAS):
        raise ValueError("La solicitud debe contener exactamente las 20 variables: " + ", ".join(COLUMNAS))
    try:
        fila = tuple(float(solicitud[c]) if solicitud[c] is not None else None for c in COLUMNAS)
    except (ValueError, TypeError) as error:
        raise ValueError("Las variables deben ser numericas o null") from error
    if any(valor is None for i, valor in enumerate(fila) if i != 4):
        raise ValueError("Solo el monto del credito puede ser null")
    if any(not np.isfinite(valor) for valor in fila if valor is not None):
        raise ValueError("No se permiten valores infinitos ni NaN")
    if fila[4] is not None and fila[4] <= 0:
        raise ValueError("El monto del credito debe ser positivo")
    imputada = imputar_columna((fila,), 4, modelo["mediana"])
    normalizada = normalizar_min_max(imputada, 4, modelo["minimo"], modelo["maximo"])
    x = np.asarray(normalizada, dtype=float)
    for indice, valores in zip((0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 17, 18, 19), modelo["categorias"]):
        if x[0, indice] not in valores:
            raise ValueError(f"Categoria desconocida en {COLUMNAS[indice]}: {x[0, indice]}")
    x_final = unir_caracteristicas(
        normalizar_numericas(x, modelo["minimos"], modelo["maximos"]),
        codificar_one_hot(x, modelo["categorias"]),
    )
    return int(predecir_knn(modelo["x"], modelo["y"], x_final, k=5)[0])


def main():
    """Lee solicitudes JSON y muestra los resultados del modelo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archivo", nargs="?", default="solicitudes_ejemplo.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
    modelo = entrenar_modelo()
    with Path(args.archivo).open(encoding="utf-8") as archivo:
        solicitudes = json.load(archivo)
    if not isinstance(solicitudes, list):
        raise ValueError("El JSON debe contener una lista de solicitudes")
    for numero, solicitud in enumerate(solicitudes, 1):
        nombre = solicitud.get("nombre", f"Solicitud {numero}")
        datos = {clave: valor for clave, valor in solicitud.items() if clave != "nombre"}
        prediccion = clasificar_solicitud(modelo, datos)
        estado = "BUEN CREDITO (clase 1)" if prediccion == 1 else "MAL CREDITO (clase 0)"
        logger.info("%s -> %s", nombre, estado)
        logger.info("Resultado educativo: requiere revision humana antes de decidir")


if __name__ == "__main__":
    main()
