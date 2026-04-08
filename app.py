"""
app.py
Flask API that loads the trained model and exposes endpoints
to classify and manage support tickets in memory.
"""

import os
import pickle
from datetime import datetime
from itertools import count

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from preprocessor import preprocess, preprocess_ticket
from naive_bayes import NaiveBayesClassifier

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend

MODEL_PATH = 'model.pkl'
ticket_sequence = count(1)
ticket_store: list[dict] = []
ticket_index: dict[str, dict] = {}

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


def extract_ticket_fields(data: dict) -> tuple[str, str, str, str]:
    """Normalize the incoming payload fields."""
    subject = str(data.get('subject', '')).strip()
    description = str(data.get('description', '')).strip()
    channel = str(data.get('channel', 'Web Portal')).strip() or 'Web Portal'
    text = str(data.get('text', '')).strip()
    return subject, description, channel, text


def generate_ticket_id() -> str:
    """Generate a deterministic ticket id for the current server session."""
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    sequence = next(ticket_sequence)
    return f'TCK-{stamp}-{sequence:04d}'


def classify_ticket_payload(data: dict) -> dict:
    """Run the same classification logic for /classify and /tickets."""
    subject, description, channel, text = extract_ticket_fields(data)

    combined_text = text or " ".join(
        part for part in (subject, description) if part
    ).strip()
    if not combined_text:
        raise ValueError(
            'Provide "text" or a combination of "subject" and/or "description"'
        )

    if subject or description:
        tokens = preprocess_ticket(subject, description, subject_weight=1)
    else:
        tokens = preprocess(text)

    if not tokens:
        raise ValueError('The text does not contain valid words after preprocessing')

    category, _ = model.predict_one(tokens)
    probabilities = model.predict_proba(tokens)
    sorted_probabilities = dict(
        sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    )

    return {
        'subject': subject,
        'description': description,
        'channel': channel,
        'normalized_text': ' '.join(tokens),
        'category': category,
        'probabilities': sorted_probabilities,
        'tokens_used': len(tokens),
    }


def build_ticket_summary(ticket: dict) -> dict:
    """Return the compact representation used by the lower UI panel."""
    return {
        'ticket_id': ticket['ticket_id'],
        'subject': ticket['subject'],
        'channel': ticket['channel'],
        'category': ticket['category'],
        'created_at': ticket['created_at'],
        'tokens_used': ticket['tokens_used'],
    }


def create_ticket(data: dict) -> dict:
    """Create, classify, and store a ticket in memory."""
    classification = classify_ticket_payload(data)
    ticket_id = generate_ticket_id()
    created_at = datetime.now().isoformat(timespec='seconds')

    ticket = {
        'ticket_id': ticket_id,
        'created_at': created_at,
        **classification,
    }

    ticket_store.append(ticket)
    ticket_index[ticket_id] = ticket
    return ticket


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
        'ticket_count': len(ticket_store),
    })


@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({'error': 'A valid JSON payload is required'}), 400

    try:
        classification = classify_ticket_payload(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'ticket_id': str(data.get('ticket_id', '')).strip(),
        **classification,
    })


@app.route('/tickets', methods=['GET'])
def list_tickets():
    tickets = [build_ticket_summary(ticket) for ticket in reversed(ticket_store)]
    return jsonify({
        'count': len(tickets),
        'tickets': tickets,
    })


@app.route('/tickets', methods=['POST'])
def create_ticket_endpoint():
    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({'error': 'A valid JSON payload is required'}), 400

    try:
        ticket = create_ticket(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify(ticket), 201


@app.route('/tickets/<ticket_id>', methods=['GET'])
def get_ticket(ticket_id: str):
    ticket = ticket_index.get(ticket_id)
    if ticket is None:
        return jsonify({'error': f'Ticket "{ticket_id}" was not found'}), 404
    return jsonify(ticket)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
