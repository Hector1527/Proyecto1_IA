"""
evaluator.py
K-Folds Cross Validation implementado manualmente.
Cálculo de Precisión, Recall, F1-Score, Accuracy y Matriz de Confusión.
"""

import math
from collections import defaultdict


# ------------------------------------------------------------------
# K-FOLDS CROSS VALIDATION
# ------------------------------------------------------------------

def k_folds_split(n: int, k: int) -> list[tuple[list[int], list[int]]]:
    """
    Divide los índices 0..n-1 en K folds.
    Devuelve lista de (train_indices, val_indices).
    """
    indices = list(range(n))
    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        # El último fold toma el resto si n no es divisible entre k
        end = start + fold_size if i < k - 1 else n
        val_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        folds.append((train_indices, val_indices))

    return folds


# ------------------------------------------------------------------
# MÉTRICAS
# ------------------------------------------------------------------

def confusion_matrix(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    """
    Construye la matriz de confusión.
    Retorna dict[true_class][pred_class] = count
    """
    matrix = {cls: {c: 0 for c in classes} for cls in classes}
    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1
    return matrix


def compute_metrics(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    """
    Calcula Precisión, Recall y F1-Score por clase,
    Accuracy global y Macro F1.

    Fórmulas:
        Precisión(c) = TP(c) / (TP(c) + FP(c))
        Recall(c)    = TP(c) / (TP(c) + FN(c))
        F1(c)        = 2 * P(c) * R(c) / (P(c) + R(c))
        Accuracy     = sum(TP) / total
        Macro F1     = mean(F1 por clase)
    """
    cm = confusion_matrix(y_true, y_pred, classes)
    metrics = {}

    total_correct = 0
    total = len(y_true)

    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in classes if other != cls)
        fn = sum(cm[cls][other] for other in classes if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        metrics[cls] = {
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1':        round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn,
        }
        total_correct += tp

    accuracy = total_correct / total if total > 0 else 0.0
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(classes)

    return {
        'per_class': metrics,
        'accuracy':  round(accuracy, 4),
        'macro_f1':  round(macro_f1, 4),
        'confusion_matrix': cm,
    }


def average_metrics(fold_results: list[dict], classes: list[str]) -> dict:
    """
    Promedia las métricas de todos los folds de K-Folds.
    """
    avg = {
        'per_class': {cls: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                      for cls in classes},
        'accuracy': 0.0,
        'macro_f1': 0.0,
        'std_accuracy': 0.0,
        'std_macro_f1': 0.0,
    }

    k = len(fold_results)
    accuracies = [r['accuracy'] for r in fold_results]
    macro_f1s  = [r['macro_f1'] for r in fold_results]

    for result in fold_results:
        avg['accuracy']  += result['accuracy']
        avg['macro_f1']  += result['macro_f1']
        for cls in classes:
            for metric in ('precision', 'recall', 'f1'):
                avg['per_class'][cls][metric] += result['per_class'][cls][metric]

    avg['accuracy']  = round(avg['accuracy'] / k, 4)
    avg['macro_f1']  = round(avg['macro_f1'] / k, 4)

    # Desviación estándar (análisis de varianza entre folds)
    avg['std_accuracy'] = round(_std(accuracies), 4)
    avg['std_macro_f1'] = round(_std(macro_f1s), 4)

    for cls in classes:
        for metric in ('precision', 'recall', 'f1'):
            avg['per_class'][cls][metric] = round(
                avg['per_class'][cls][metric] / k, 4
            )

    return avg


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def print_report(results: dict, classes: list[str], fold_results: list[dict] = None):
    """Imprime un reporte legible en consola."""
    print("\n" + "=" * 60)
    print("REPORTE DE EVALUACIÓN")
    print("=" * 60)

    print(f"\nAccuracy global : {results['accuracy']:.4f}")
    print(f"Macro F1        : {results['macro_f1']:.4f}")
    if 'std_accuracy' in results:
        print(f"Std Accuracy    : {results['std_accuracy']:.4f}")
        print(f"Std Macro F1    : {results['std_macro_f1']:.4f}")

    print("\nMétricas por clase:")
    print(f"{'Clase':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    for cls in classes:
        m = results['per_class'][cls]
        print(f"{cls:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    if 'confusion_matrix' in results:
        print("\nMatriz de Confusión (filas=real, columnas=predicho):")
        cm = results['confusion_matrix']
        header = f"{'':>20}" + "".join(f"{c[:8]:>10}" for c in classes)
        print(header)
        for cls in classes:
            row = f"{cls[:20]:<20}" + "".join(
                f"{cm[cls][c]:>10}" for c in classes
            )
            print(row)

    if fold_results:
        print("\nAccuracy por fold:")
        for i, r in enumerate(fold_results, 1):
            print(f"  Fold {i}: {r['accuracy']:.4f}  |  Macro F1: {r['macro_f1']:.4f}")
