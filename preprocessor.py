"""
preprocessor.py
Cleaning, tokenization, stopword removal, stemming,
and optional bigram generation.
"""

import re
import string

try:
    import nltk
except Exception:
    nltk = None

# Fallback stopwords in case NLTK resources are not available locally
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

if nltk is not None:
    try:
        nltk.data.find('corpora/stopwords')
        from nltk.corpus import stopwords as nltk_sw
        _stop_words = set(nltk_sw.words('english'))
    except Exception:
        _stop_words = _FALLBACK_STOPWORDS
else:
    _stop_words = _FALLBACK_STOPWORDS

# Keep negations and useful support-ticket words.
_protected_words = {'no', 'nor', 'not', 'need', 'help', 'want', 'would', 'could'}
_stop_words = set(_stop_words) - _protected_words

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    _use_stemmer = True
except Exception:
    _use_stemmer = False


def clean_text(text: str) -> str:
    """
    Clean raw text:
    - Remove placeholders such as {product_purchased}
    - Convert to lowercase
    - Remove URLs, emails, numbers, and punctuation
    """
    if not isinstance(text, str):
        return ""

    # Remove placeholders such as {variable}
    text = re.sub(r'\{[^}]+\}', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Convert to lowercase
    text = text.lower()

    # Expand common negations before removing punctuation
    text = re.sub(r"can't", ' can not ', text)
    text = re.sub(r"won't", ' will not ', text)
    text = re.sub(r"n't", ' not ', text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace and line breaks
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text: str, add_bigrams: bool = False) -> list[str]:
    """
    Tokenize clean text, remove stopwords, and apply stemming.
    Optionally add bigrams to capture frequent phrases.
    Returns the processed token list.
    """
    tokens = text.split()

    # Remove stopwords and very short tokens
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]

    # Stemming
    if _use_stemmer:
        tokens = [_stemmer.stem(t) for t in tokens]

    if add_bigrams:
        tokens = tokens + build_bigrams(tokens)

    return tokens


def build_bigrams(tokens: list[str]) -> list[str]:
    """Generate simple bigrams from a token sequence."""
    return [f'{left}__{right}' for left, right in zip(tokens, tokens[1:])]


def preprocess(text: str, add_bigrams: bool = True) -> list[str]:
    """Complete free-text pipeline: cleaning + tokenization."""
    return tokenize(clean_text(text), add_bigrams=add_bigrams)


def preprocess_ticket(
    subject: str,
    description: str,
    subject_weight: int = 1,
    add_bigrams: bool = True,
) -> list[str]:
    """
    Preprocess a ticket using subject and description separately.
    The subject can be repeated to increase its semantic weight.
    """
    subject_tokens = tokenize(clean_text(subject), add_bigrams=False)
    description_tokens = tokenize(clean_text(description), add_bigrams=False)

    tokens = (subject_tokens * max(subject_weight, 1)) + description_tokens
    if add_bigrams:
        tokens = tokens + build_bigrams(tokens)
    return tokens
