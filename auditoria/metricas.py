"""Evalua predicciones con la clase cero como positiva."""
import numpy as np

def calcular_matriz_confusion(y_real, y_pred):
    """Calcular matriz confusion.

    Args:
        y_real: Parametro de entrada.
        y_pred: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    if y_real.shape != y_pred.shape:
        raise ValueError('Las etiquetas reales y predichas deben tener las mismas dimensiones')
    vp = np.sum((y_real == 0) & (y_pred == 0))
    fn = np.sum((y_real == 0) & (y_pred == 1))
    fp = np.sum((y_real == 1) & (y_pred == 0))
    vn = np.sum((y_real == 1) & (y_pred == 1))
    matriz = np.array([[vp, fn], [fp, vn]])
    return matriz

def calcular_metricas(matriz):
    """Calcular metricas.

    Args:
        matriz: Parametro de entrada.

    Returns:
        Resultado de la operacion.
"""
    vp = matriz[0, 0]
    fn = matriz[0, 1]
    fp = matriz[1, 0]
    vn = matriz[1, 1]
    total = np.sum(matriz)
    exactitud = (vp + vn) / total if total > 0 else 0.0
    precision = vp / (vp + fp) if vp + fp > 0 else 0.0
    recall = vp / (vp + fn) if vp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {'exactitud': exactitud, 'precision': precision, 'recall': recall, 'f1': f1}
