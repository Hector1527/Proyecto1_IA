"""
app.py
API Flask que carga el modelo entrenado y expone un endpoint
para clasificar tickets nuevos.

Endpoints:
  POST /classify  { "text": "..." }  → { "category": "...", "probabilities": {...} }
  GET  /health    → { "status": "ok", "vocabulary_size": N, "classes": [...] }
"""

import pickle
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from preprocessor import preprocess
from naive_bayes import NaiveBayesClassifier

app = Flask(__name__)
CORS(app)  # Permite peticiones desde el frontend HTML

MODEL_PATH = 'model.pkl'

# ------------------------------------------------------------------
# CARGA DEL MODELO AL INICIAR
# ------------------------------------------------------------------

def load_model() -> NaiveBayesClassifier:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{MODEL_PATH}'. "
            "Ejecuta primero: python train.py"
        )
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)

    model = NaiveBayesClassifier()
    model.from_dict(data)
    print(f"[OK] Modelo cargado — {len(model.vocabulary):,} palabras en vocabulario.")
    return model


model = load_model()


# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------

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

    if not data or 'text' not in data:
        return jsonify({'error': 'Se requiere el campo "text"'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'El texto no puede estar vacío'}), 400

    # Preprocesar
    tokens = preprocess(text)

    if not tokens:
        return jsonify({'error': 'El texto no contiene palabras válidas tras el preprocesamiento'}), 400

    # Clasificar
    category, _ = model.predict_one(tokens)
    probabilities = model.predict_proba(tokens)

    # Ordenar probabilidades de mayor a menor
    sorted_probs = dict(
        sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    )

    return jsonify({
        'category':      category,
        'probabilities': sorted_probs,
        'tokens_used':   len(tokens),
    })


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
