# Cambios de calidad y demostracion KNN

- Se mantuvo la logica original del proyecto, incluida la semilla 42 y k=5.
- Los modulos tienen documentacion inicial y funciones con docstrings; el calculo de distancias usa formato NumPy.
- `main.py` usa `logging` en lugar de `print`.
- Los archivos `.flake8`, `pyproject.toml` y `requirements-calidad.txt` permiten instalar y ejecutar las herramientas de estilo.
- `demostracion_creditos.py` carga el conjunto de entrenamiento, calcula parametros solo con entrenamiento y clasifica solicitudes JSON sin incorporarlas al ajuste.
- Los dos casos de `solicitudes_ejemplo.json` son ejemplos didacticos derivados de perfiles del dataset; no son clientes reales ni garantizan aprobacion bancaria.

## Ejecucion

```powershell
python main.py
python demostracion_creditos.py
python demostracion_creditos.py mis_solicitudes.json
python -m pip install -r requirements-calidad.txt
python -m black --check .
python -m flake8 .
python -m isort --check-only .
```

## Notas

El modelo KNN se reconstruye al iniciar, con la misma semilla y el mismo dataset: no existe un archivo de modelo serializado. Los campos de la solicitud deben conservar las 20 variables originales, en su codificacion numerica. Las predicciones son resultados del modelo, no decisiones reales de otorgamiento.
