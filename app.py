"""
app.py
Flask API that loads the trained model and exposes endpoints
to classify new support tickets.

Endpoints:
  POST /classify  { "text": "..." }  -> { "category": "...", "probabilities": {...} }
  GET  /health    -> { "status": "ok", "vocabulary_size": N, "classes": [...] }
"""

import pickle
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from preprocessor import preprocess, preprocess_ticket
from naive_bayes import NaiveBayesClassifier

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend

MODEL_PATH = 'model.pkl'

# ------------------------------------------------------------------
# LOAD MODEL AT STARTUP
# ------------------------------------------------------------------

def load_model() -> NaiveBayesClassifier:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Run this first: python train.py"
        )
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)

    model = NaiveBayesClassifier()
    model.from_dict(data)
    print(f"[OK] Model loaded - {len(model.vocabulary):,} words in vocabulary.")
    return model


model = load_model()


def build_ticket_text(data: dict) -> tuple[str, str, str]:
    """
    Normalize the ticket payload.
    Supports:
      - text
      - subject + description
    """
    ticket_id = str(data.get('ticket_id', '')).strip()
    subject = str(data.get('subject', '')).strip()
    description = str(data.get('description', '')).strip()
    text = str(data.get('text', '')).strip()

    if text:
        return ticket_id, subject, text

    combined = " ".join(part for part in (subject, description) if part).strip()
    return ticket_id, subject, combined


# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'vocabulary_size': len(model.vocabulary),
        'classes': model.classes,
    })


@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json(force=True)

    if not data:
        return jsonify({'error': 'A valid JSON payload is required'}), 400

    ticket_id, subject, text = build_ticket_text(data)
    if not text:
        return jsonify({
            'error': 'Provide "text" or a combination of "subject" and/or "description"'
        }), 400

    # Preprocess using the same logic as training
    if subject or data.get('description'):
        tokens = preprocess_ticket(
            subject,
            str(data.get('description', '')).strip(),
            subject_weight=1,
        )
    else:
        tokens = preprocess(text)

    if not tokens:
        return jsonify({'error': 'The text does not contain valid words after preprocessing'}), 400

    # Classify
    category, _ = model.predict_one(tokens)
    probabilities = model.predict_proba(tokens)

    # Sort probabilities from highest to lowest
    sorted_probs = dict(
        sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    )

    return jsonify({
        'ticket_id':     ticket_id,
        'subject':       subject,
        'normalized_text': text,
        'category':      category,
        'probabilities': sorted_probs,
        'tokens_used':   len(tokens),
    })


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
