import re
import unicodedata

from nltk.corpus import stopwords
from nltk import stem


BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z]')
DUPLICATED_LETTERS = re.compile(r'([a-z])\1{2,}')
STOPWORDS = set(stopwords.words('portuguese'))
stemmer = stem.RSLPStemmer()


def strip_accents(text: str) -> str:
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError):
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")

    return str(text)


def preprocess_text(text: str) -> str:
    text = strip_accents(text)
    text = text.lower()
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = ' '.join(word for word in text.split())

    return text


def preprocess_text_stop(text: str) -> str:
    text = text.lower()

    text = strip_accents(text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = ' '.join(word for word in text.split()
                    if word not in STOPWORDS and len(word) > 2)

    return text


def preprocess_stemming_text(text: str) -> str:
    text = text.lower()

    text = strip_accents(text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = ' '.join(stemmer.stem(word)
                    for word in text.split() if word not in STOPWORDS and len(word) > 2)

    return text


def preprocess_stemming_text_duplicated(text: str) -> str:
    text = text.lower()

    text = strip_accents(text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = DUPLICATED_LETTERS.sub(r'\1', text)
    text = ' '.join(stemmer.stem(word)
                    for word in text.split() if word not in STOPWORDS and len(word) > 2)

    return text
