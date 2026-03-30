"""
naive_bayes.py
Implementación manual de Naïve Bayes Multinomial con:
  - Bag of Words
  - Laplace Smoothing
  - Suma de logaritmos para evitar underflow
"""

import math
from collections import defaultdict


class NaiveBayesClassifier:

    def __init__(self):
        self.vocabulary = set()          # Vocabulario del corpus
        self.class_priors = {}           # log P(clase)
        self.word_log_probs = {}         # log P(palabra | clase)
        self.classes = []                # Lista de clases únicas
        self._class_word_counts = {}     # Conteo de palabras por clase
        self._class_total_words = {}     # Total de palabras por clase

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------

    def fit(self, documents: list[list[str]], labels: list[str]):
        """
        Entrena el modelo dado un corpus ya preprocesado.

        Args:
            documents: Lista de documentos, cada uno como lista de tokens.
            labels:    Lista de etiquetas correspondientes.
        """
        self.classes = list(set(labels))
        n_docs = len(documents)

        # --- Contar documentos por clase (para prior) ---
        class_doc_counts = defaultdict(int)
        for label in labels:
            class_doc_counts[label] += 1

        # --- Calcular log P(clase) ---
        self.class_priors = {
            cls: math.log(count / n_docs)
            for cls, count in class_doc_counts.items()
        }

        # --- Construir vocabulario y contar palabras por clase ---
        self._class_word_counts = {cls: defaultdict(int) for cls in self.classes}
        self._class_total_words = {cls: 0 for cls in self.classes}

        for doc, label in zip(documents, labels):
            for word in doc:
                self.vocabulary.add(word)
                self._class_word_counts[label][word] += 1
                self._class_total_words[label] += 1

        vocab_size = len(self.vocabulary)

        # --- Calcular log P(palabra | clase) con Laplace Smoothing ---
        # P(w|c) = (count(w,c) + 1) / (total_words(c) + |V|)
        self.word_log_probs = {}
        for cls in self.classes:
            self.word_log_probs[cls] = {}
            total = self._class_total_words[cls]
            for word in self.vocabulary:
                count = self._class_word_counts[cls].get(word, 0)
                self.word_log_probs[cls][word] = math.log(
                    (count + 1) / (total + vocab_size)
                )

            # Probabilidad para palabras fuera del vocabulario (OOV)
            self.word_log_probs[cls]['<OOV>'] = math.log(
                1 / (total + vocab_size)
            )

    # ------------------------------------------------------------------
    # INFERENCIA
    # ------------------------------------------------------------------

    def predict_one(self, tokens: list[str]) -> tuple[str, dict]:
        """
        Clasifica un documento ya preprocesado.

        Returns:
            (clase_predicha, dict con log-scores por clase)
        """
        scores = {}
        for cls in self.classes:
            # Empezar con el prior
            score = self.class_priors[cls]
            # Sumar log-verosimilitudes (evita underflow numérico)
            for word in tokens:
                if word in self.vocabulary:
                    score += self.word_log_probs[cls][word]
                else:
                    score += self.word_log_probs[cls]['<OOV>']
            scores[cls] = score

        predicted = max(scores, key=scores.get)
        return predicted, scores

    def predict(self, documents: list[list[str]]) -> list[str]:
        """Clasifica una lista de documentos."""
        return [self.predict_one(doc)[0] for doc in documents]

    def predict_proba(self, tokens: list[str]) -> dict:
        """
        Devuelve probabilidades normalizadas por clase (para mostrar en UI).
        Convierte log-scores a probabilidades usando softmax numérico.
        """
        _, log_scores = self.predict_one(tokens)

        # Estabilidad numérica: restar el máximo antes de exp
        max_score = max(log_scores.values())
        exp_scores = {cls: math.exp(s - max_score) for cls, s in log_scores.items()}
        total = sum(exp_scores.values())
        return {cls: round(v / total, 4) for cls, v in exp_scores.items()}

    # ------------------------------------------------------------------
    # SERIALIZACIÓN (para guardar / cargar el modelo)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serializa el modelo a un diccionario (para JSON o pickle)."""
        return {
            'vocabulary': list(self.vocabulary),
            'class_priors': self.class_priors,
            'word_log_probs': self.word_log_probs,
            'classes': self.classes,
            '_class_word_counts': {
                cls: dict(counts)
                for cls, counts in self._class_word_counts.items()
            },
            '_class_total_words': self._class_total_words,
        }

    def from_dict(self, data: dict):
        """Carga el modelo desde un diccionario."""
        self.vocabulary = set(data['vocabulary'])
        self.class_priors = data['class_priors']
        self.word_log_probs = data['word_log_probs']
        self.classes = data['classes']
        self._class_word_counts = {
            cls: defaultdict(int, counts)
            for cls, counts in data['_class_word_counts'].items()
        }
        self._class_total_words = data['_class_total_words']
        return self
