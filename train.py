"""
train.py
Script de entrenamiento completo:
  1. Carga y preprocesa el dataset
  2. Ejecuta K-Folds Cross Validation (K=5)
  3. Entrena el modelo final con todos los datos
  4. Guarda el modelo en model.pkl
"""

import pickle
import random
import pandas as pd

from preprocessor import preprocess
from naive_bayes import NaiveBayesClassifier
from evaluator import k_folds_split, compute_metrics, average_metrics, print_report

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
DATASET_PATH = 'customer_support_tickets.csv'   # <-- Cambia este nombre si tu CSV se llama diferente
MODEL_PATH   = 'model.pkl'
K            = 5
RANDOM_SEED  = 42

# Columnas del dataset (ajusta si tu CSV las llama diferente)
TEXT_COLS  = ['Ticket Subject', 'Ticket Description']  # Se concatenan
LABEL_COL  = 'Ticket Type'

# Clases esperadas (según el proyecto)
CLASSES = ['Technical issue', 'Billing inquiry', 'Product inquiry',
           'Refund request', 'Cancellation request']


# ------------------------------------------------------------------
# CARGA Y PREPROCESAMIENTO
# ------------------------------------------------------------------

def load_data(path: str):
    print(f"[1/4] Cargando dataset desde '{path}'...")
    df = pd.read_csv(path)

    print(f"      Filas totales: {len(df)}")
    print(f"      Distribución de clases:\n{df[LABEL_COL].value_counts()}\n")

    # Eliminar filas con texto o etiqueta vacíos
    df = df.dropna(subset=[LABEL_COL])
    for col in TEXT_COLS:
        df[col] = df[col].fillna('')

    # Concatenar columnas de texto
    df['full_text'] = df[TEXT_COLS].apply(lambda row: ' '.join(row.values), axis=1)

    # Filtrar solo las clases esperadas (por si el dataset tiene otras)
    df = df[df[LABEL_COL].isin(CLASSES)].reset_index(drop=True)
    print(f"      Filas después de filtrar: {len(df)}\n")

    return df['full_text'].tolist(), df[LABEL_COL].tolist()


def preprocess_corpus(texts: list[str]) -> list[list[str]]:
    print("[2/4] Preprocesando textos...")
    processed = [preprocess(t) for t in texts]
    avg_tokens = sum(len(d) for d in processed) / len(processed)
    print(f"      Promedio de tokens por documento: {avg_tokens:.1f}\n")
    return processed


# ------------------------------------------------------------------
# K-FOLDS CROSS VALIDATION
# ------------------------------------------------------------------

def run_k_folds(documents, labels, k=K):
    print(f"[3/4] Ejecutando {k}-Folds Cross Validation...")
    classes = list(set(labels))

    # Mezclar los datos de forma reproducible
    random.seed(RANDOM_SEED)
    combined = list(zip(documents, labels))
    random.shuffle(combined)
    documents, labels = zip(*combined)
    documents, labels = list(documents), list(labels)

    folds = k_folds_split(len(documents), k)
    fold_results = []

    for i, (train_idx, val_idx) in enumerate(folds, 1):
        train_docs   = [documents[j] for j in train_idx]
        train_labels = [labels[j]    for j in train_idx]
        val_docs     = [documents[j] for j in val_idx]
        val_labels   = [labels[j]    for j in val_idx]

        model = NaiveBayesClassifier()
        model.fit(train_docs, train_labels)
        predictions = model.predict(val_docs)

        result = compute_metrics(val_labels, predictions, classes)
        fold_results.append(result)
        print(f"  Fold {i}/{k} — Accuracy: {result['accuracy']:.4f} | Macro F1: {result['macro_f1']:.4f}")

    avg = average_metrics(fold_results, classes)
    print_report(avg, classes, fold_results)

    return fold_results, avg, documents, labels


# ------------------------------------------------------------------
# ENTRENAMIENTO FINAL Y GUARDADO
# ------------------------------------------------------------------

def train_final_model(documents, labels):
    print("\n[4/4] Entrenando modelo final con todos los datos...")
    model = NaiveBayesClassifier()
    model.fit(documents, labels)
    print(f"      Vocabulario: {len(model.vocabulary):,} palabras")
    print(f"      Clases: {model.classes}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model.to_dict(), f)
    print(f"      Modelo guardado en '{MODEL_PATH}'\n")

    return model


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == '__main__':
    texts, labels = load_data(DATASET_PATH)
    documents     = preprocess_corpus(texts)
    fold_results, avg_metrics, documents, labels = run_k_folds(documents, labels)
    model = train_final_model(documents, labels)

    print("✓ Entrenamiento completado exitosamente.")
    print(f"  Accuracy promedio (K-Folds): {avg_metrics['accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    print(f"  Macro F1 promedio (K-Folds): {avg_metrics['macro_f1']:.4f} ± {avg_metrics['std_macro_f1']:.4f}")
