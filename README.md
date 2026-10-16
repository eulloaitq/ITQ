# Proyecto de Riesgo Crediticio con KNN

## Descripción

Este proyecto desarrolla un sistema de clasificación de riesgo crediticio mediante el algoritmo K-Nearest Neighbors (KNN), implementado con Python y NumPy.

El objetivo es aplicar conceptos de programación funcional, procesamiento de datos, operaciones vectorizadas y evaluación de modelos de aprendizaje automático, sin utilizar bibliotecas de clasificación como Scikit-learn.

Se utiliza el conjunto de datos **South German Credit**, que contiene información de 1.000 solicitudes de crédito.

## Objetivos

- Implementar un proceso de ingesta y limpieza de datos mediante programación funcional.
- Construir un clasificador KNN utilizando operaciones vectorizadas de NumPy.
- Evaluar el rendimiento del clasificador mediante una matriz de confusión y métricas de clasificación.
- Realizar una auditoría de equidad utilizando el indicador Disparate Impact.

## Tecnologías utilizadas

- Python 3.12
- NumPy 2.5.3
- Visual Studio Code
- Entorno virtual de Python (`venv`)

## Metodología

### Hito 1: Ingesta y procesamiento funcional

Se implementaron procedimientos para:

- Lectura del dataset y almacenamiento en estructuras inmutables.
- Verificación de valores ausentes.
- Simulación de valores ausentes para comprobar el proceso de imputación.
- Imputación mediante la mediana.
- Detección y eliminación de valores atípicos mediante el método IQR.
- Normalización Min-Max.
- Construcción de un pipeline funcional.

El conjunto original contiene 1.000 registros y 21 columnas, incluida la variable objetivo.

### Hito 2: Clasificador KNN vectorizado

Se desarrolló un clasificador KNN sin utilizar Scikit-learn.

El procedimiento incluye:

- División reproducible de los datos en entrenamiento (80 %) y prueba (20 %).
- Preprocesamiento utilizando parámetros obtenidos exclusivamente del conjunto de entrenamiento.
- Codificación One-Hot de variables categóricas.
- Normalización de variables numéricas.
- Cálculo vectorizado de distancias euclidianas.
- Identificación de los K vecinos más cercanos.
- Predicción mediante votación mayoritaria.

Se utilizó **K = 5**.

Después de aplicar el filtro IQR al entrenamiento, se conservaron 732 registros de entrenamiento. El conjunto de prueba mantuvo sus 200 registros.

La matriz final contiene 62 características.

### Hito 3: Evaluación y auditoría

Se implementó una matriz de confusión y se calcularon las métricas de exactitud, precisión, recall y F1-Score.

Para el análisis de riesgo se considera positiva la clase 0 (mal crédito).

También se realizó una auditoría por grupos de edad:

- Grupo A: menores de 25 años.
- Grupo B: personas de 25 años o más.

Se analizaron las tasas de clasificación favorable, la exactitud y los falsos negativos de cada grupo.

## Resultados del modelo

| Indicador | Resultado |
|---|---:|
| Registros evaluados | 200 |
| Exactitud | 66,50 % |
| Precisión | 47,06 % |
| Recall | 24,62 % |
| F1-Score | 32,32 % |
| Verdaderos positivos | 16 |
| Falsos negativos | 49 |
| Falsos positivos | 18 |
| Verdaderos negativos | 117 |

## Resultados de la auditoría

| Indicador | Resultado |
|---|---:|
| Disparate Impact | 0,9092 |
| Tasa favorable, menores de 25 años | 76,19 % |
| Tasa favorable, 25 años o más | 83,80 % |
| Exactitud, menores de 25 años | 71,43 % |
| Exactitud, 25 años o más | 65,92 % |
| Tasa de falsos negativos, menores de 25 años | 55,56 % |
| Tasa de falsos negativos, 25 años o más | 78,57 % |

## Instalación y ejecución

1. Abrir una terminal en la carpeta principal del proyecto.
2. Crear el entorno virtual:

   ```powershell
   python -m venv .venv
   ```

3. Activarlo en PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Instalar las dependencias:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Ejecutar el programa:

   ```powershell
   python main.py
   ```

Si PowerShell impide activar el entorno virtual, se puede permitir temporalmente la ejecución de scripts durante la sesión actual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Conclusiones

El proyecto demuestra que es posible implementar un clasificador KNN mediante operaciones vectorizadas de NumPy y aplicar programación funcional en el procesamiento de datos.

El modelo obtuvo una exactitud del 66,50 %. Sin embargo, el recall del 24,62 % evidencia dificultades para identificar los casos de mal crédito, por lo que no se recomienda utilizarlo para tomar decisiones crediticias reales sin mejoras y validaciones adicionales.

La auditoría obtuvo un Disparate Impact de 0,9092. Aunque este valor no indica una disparidad marcada según la regla orientativa del 80 %, se encontraron diferencias en las tasas de falsos negativos entre los grupos de edad.

Estos resultados demuestran la importancia de evaluar tanto el rendimiento predictivo como los posibles efectos desiguales de un modelo de aprendizaje automático.

## Dataset

South German Credit, disponible en el repositorio de aprendizaje automático de la Universidad de California, Irvine (UCI Machine Learning Repository).