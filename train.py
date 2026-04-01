"""
train.py
Full training script:
  1. Load and preprocess the dataset
  2. Run K-Folds cross validation (K=5)
  3. Train the final model on all data
  4. Save the model to model.pkl
"""

import pickle
import random
import pandas as pd

from preprocessor import preprocess_ticket
from naive_bayes import NaiveBayesClassifier
from evaluator import stratified_k_folds_split, compute_metrics, average_metrics, print_report

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
DATASET_PATH = 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
MODEL_PATH   = 'model.pkl'
K            = 5
RANDOM_SEED  = 42
SUBJECT_WEIGHT = 1
MIN_TOKEN_FREQUENCY = 1

# Dataset columns
TEXT_COLS    = ['instruction']
LABEL_COL    = 'category'

# Expected classes from the dataset
CLASSES = None


# ------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# ------------------------------------------------------------------

def load_data(path: str):
    print(f"[1/4] Loading dataset from '{path}'...")
    df = pd.read_csv(path)

    print(f"      Total rows: {len(df)}")
    print(f"      Class distribution:\n{df[LABEL_COL].value_counts()}\n")

    # Remove rows with missing label
    df = df.dropna(subset=[LABEL_COL])
    df[TEXT_COLS[0]] = df[TEXT_COLS[0]].fillna('')

    if CLASSES:
        df = df[df[LABEL_COL].isin(CLASSES)].reset_index(drop=True)
    print(f"      Rows after filtering: {len(df)}\n")

    return (
        [''] * len(df),
        df[TEXT_COLS[0]].tolist(),
        df[LABEL_COL].tolist(),
    )


def preprocess_corpus(subjects: list[str], descriptions: list[str]) -> list[list[str]]:
    print("[2/4] Preprocessing text...")
    processed = [
        preprocess_ticket(subject, description, subject_weight=SUBJECT_WEIGHT)
        for subject, description in zip(subjects, descriptions)
    ]
    avg_tokens = sum(len(d) for d in processed) / len(processed)
    print(f"      Average tokens per document: {avg_tokens:.1f}\n")
    return processed


# ------------------------------------------------------------------
# K-FOLDS CROSS VALIDATION
# ------------------------------------------------------------------

def run_k_folds(documents, labels, k=K):
    print(f"[3/4] Running {k}-Folds Cross Validation...")
    classes = list(set(labels))

    # Shuffle data reproducibly
    random.seed(RANDOM_SEED)
    combined = list(zip(documents, labels))
    random.shuffle(combined)
    documents, labels = zip(*combined)
    documents, labels = list(documents), list(labels)

    folds = stratified_k_folds_split(labels, k, seed=RANDOM_SEED)
    fold_results = []

    for i, (train_idx, val_idx) in enumerate(folds, 1):
        train_docs   = [documents[j] for j in train_idx]
        train_labels = [labels[j]    for j in train_idx]
        val_docs     = [documents[j] for j in val_idx]
        val_labels   = [labels[j]    for j in val_idx]

        model = NaiveBayesClassifier(min_token_frequency=MIN_TOKEN_FREQUENCY)
        model.fit(train_docs, train_labels)
        predictions = model.predict(val_docs)

        result = compute_metrics(val_labels, predictions, classes)
        fold_results.append(result)
        print(f"  Fold {i}/{k} - Accuracy: {result['accuracy']:.4f} | Macro F1: {result['macro_f1']:.4f}")

    avg = average_metrics(fold_results, classes)
    print_report(avg, classes, fold_results)

    return fold_results, avg, documents, labels


# ------------------------------------------------------------------
# FINAL TRAINING AND MODEL SAVING
# ------------------------------------------------------------------

def train_final_model(documents, labels):
    print("\n[4/4] Training final model on all data...")
    model = NaiveBayesClassifier(min_token_frequency=MIN_TOKEN_FREQUENCY)
    model.fit(documents, labels)
    print(f"      Vocabulary: {len(model.vocabulary):,} words")
    print(f"      Classes: {model.classes}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model.to_dict(), f)
    print(f"      Model saved to '{MODEL_PATH}'\n")

    return model


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == '__main__':
    subjects, descriptions, labels = load_data(DATASET_PATH)
    documents     = preprocess_corpus(subjects, descriptions)
    fold_results, avg_metrics, documents, labels = run_k_folds(documents, labels)
    model = train_final_model(documents, labels)

    print("Training completed successfully.")
    print(f"  Average Accuracy (K-Folds): {avg_metrics['accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    print(f"  Average Macro F1 (K-Folds): {avg_metrics['macro_f1']:.4f} ± {avg_metrics['std_macro_f1']:.4f}")
