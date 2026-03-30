"""
preprocessor.py
Limpieza, tokenización, eliminación de stopwords y stemming.
"""

import re
import string
import nltk

# Stopwords de respaldo (por si NLTK no tiene conexión en el primer arranque)
_FALLBACK_STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
    'weren','won','wouldn','would','could','get','got','also','please','may',
    'us','like','one','even','well','still','back','go','going','want','need',
    'use','using','used','make','made','know','help','try','tried','look',
    'see','think','come','say','said','work','way','time','day','new','good',
}

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords as nltk_sw
    _stop_words = set(nltk_sw.words('english'))
except Exception:
    _stop_words = _FALLBACK_STOPWORDS

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    _use_stemmer = True
except Exception:
    _use_stemmer = False


def clean_text(text: str) -> str:
    """
    Limpia el texto crudo:
    - Elimina placeholders como {product_purchased}
    - Convierte a minúsculas
    - Elimina URLs, emails, números y puntuación
    """
    if not isinstance(text, str):
        return ""

    # Eliminar placeholders tipo {variable}
    text = re.sub(r'\{[^}]+\}', '', text)

    # Eliminar URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Eliminar emails
    text = re.sub(r'\S+@\S+', '', text)

    # Eliminar números
    text = re.sub(r'\d+', '', text)

    # Convertir a minúsculas
    text = text.lower()

    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Eliminar espacios extra y saltos de línea
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text: str) -> list[str]:
    """
    Tokeniza el texto limpio, elimina stopwords y aplica stemming.
    Devuelve lista de tokens procesados.
    """
    tokens = text.split()

    # Eliminar stopwords y tokens muy cortos
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]

    # Stemming
    if _use_stemmer:
        tokens = [_stemmer.stem(t) for t in tokens]

    return tokens


def preprocess(text: str) -> list[str]:
    """Pipeline completo: limpieza + tokenización."""
    return tokenize(clean_text(text))
