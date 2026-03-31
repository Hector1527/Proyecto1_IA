"""
naive_bayes.py
Manual Multinomial Naive Bayes implementation with:
  - Bag of Words
  - Laplace smoothing
  - Log-sum inference to avoid numerical underflow
"""

import math
from collections import defaultdict


class NaiveBayesClassifier:

    def __init__(self, min_token_frequency: int = 1):
        self.vocabulary = set()          # Corpus vocabulary
        self.class_priors = {}           # log P(class)
        self.word_log_probs = {}         # log P(word | class)
        self.classes = []                # Unique class labels
        self._class_word_counts = {}     # Word counts per class
        self._class_total_words = {}     # Total words per class
        self.min_token_frequency = max(1, min_token_frequency)

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def fit(self, documents: list[list[str]], labels: list[str]):
        """
        Train the model on a preprocessed corpus.

        Args:
            documents: List of documents, each represented as a token list.
            labels:    List of matching labels.
        """
        self.classes = list(set(labels))
        n_docs = len(documents)

        # --- Count documents per class for the prior ---
        class_doc_counts = defaultdict(int)
        for label in labels:
            class_doc_counts[label] += 1

        # --- Compute log P(class) ---
        self.class_priors = {
            cls: math.log(count / n_docs)
            for cls, count in class_doc_counts.items()
        }

        # --- Count global token frequency to filter vocabulary noise ---
        global_word_counts = defaultdict(int)
        for doc in documents:
            for word in doc:
                global_word_counts[word] += 1

        self.vocabulary = {
            word for word, count in global_word_counts.items()
            if count >= self.min_token_frequency
        }

        # --- Build per-class counts using the filtered vocabulary ---
        self._class_word_counts = {cls: defaultdict(int) for cls in self.classes}
        self._class_total_words = {cls: 0 for cls in self.classes}

        for doc, label in zip(documents, labels):
            for word in doc:
                if word in self.vocabulary:
                    self._class_word_counts[label][word] += 1
                    self._class_total_words[label] += 1

        vocab_size = len(self.vocabulary)

        # --- Compute log P(word | class) with Laplace smoothing ---
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

            # Probability for out-of-vocabulary words (OOV)
            self.word_log_probs[cls]['<OOV>'] = math.log(
                1 / (total + vocab_size)
            )

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def predict_one(self, tokens: list[str]) -> tuple[str, dict]:
        """
        Classify one preprocessed document.

        Returns:
            (predicted_class, dict with per-class log scores)
        """
        scores = {}
        for cls in self.classes:
            # Start from the prior
            score = self.class_priors[cls]
            # Sum log-likelihoods to avoid numerical underflow
            for word in tokens:
                if word in self.vocabulary:
                    score += self.word_log_probs[cls][word]
                else:
                    score += self.word_log_probs[cls]['<OOV>']
            scores[cls] = score

        predicted = max(scores, key=scores.get)
        return predicted, scores

    def predict(self, documents: list[list[str]]) -> list[str]:
        """Classify a list of documents."""
        return [self.predict_one(doc)[0] for doc in documents]

    def predict_proba(self, tokens: list[str]) -> dict:
        """
        Return normalized per-class probabilities for the UI.
        Converts log scores into probabilities using a numerically stable softmax.
        """
        _, log_scores = self.predict_one(tokens)

        # Numerical stability: subtract the max score before exp
        max_score = max(log_scores.values())
        exp_scores = {cls: math.exp(s - max_score) for cls, s in log_scores.items()}
        total = sum(exp_scores.values())
        return {cls: round(v / total, 4) for cls, v in exp_scores.items()}

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the model into a dictionary for pickle or JSON."""
        return {
            'vocabulary': list(self.vocabulary),
            'class_priors': self.class_priors,
            'word_log_probs': self.word_log_probs,
            'classes': self.classes,
            'min_token_frequency': self.min_token_frequency,
            '_class_word_counts': {
                cls: dict(counts)
                for cls, counts in self._class_word_counts.items()
            },
            '_class_total_words': self._class_total_words,
        }

    def from_dict(self, data: dict):
        """Load the model from a dictionary."""
        self.vocabulary = set(data['vocabulary'])
        self.class_priors = data['class_priors']
        self.word_log_probs = data['word_log_probs']
        self.classes = data['classes']
        self.min_token_frequency = data.get('min_token_frequency', 1)
        self._class_word_counts = {
            cls: defaultdict(int, counts)
            for cls, counts in data['_class_word_counts'].items()
        }
        self._class_total_words = data['_class_total_words']
        return self
