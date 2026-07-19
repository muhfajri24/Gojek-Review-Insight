from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.base import BaseEstimator, TransformerMixin

NEGATIONS = {"tidak", "tak", "bukan", "belum", "jangan", "gak", "ga", "nggak", "enggak"}
PROJECT_STOPWORDS = {"aplikasi", "app", "apk", "gojek", "nya", "yg", "aja", "bgt", "dong", "nih", "sih", "tau", "biar", "buat", "iya", "udah", "kak", "min"}
SLANG_MAP = {"gk": "gak", "ngga": "nggak", "engga": "enggak", "tdk": "tidak", "blm": "belum", "udh": "sudah", "tp": "tapi", "dr": "dari", "krn": "karena", "bgs": "bagus"}
STOPWORDS = (set(StopWordRemoverFactory().get_stop_words()) | PROJECT_STOPWORDS) - NEGATIONS
STEMMER = StemmerFactory().create_stemmer()


@lru_cache(maxsize=100000)
def stem_token(token: str) -> str:
    return STEMMER.stem(token)


@lru_cache(maxsize=100000)
def basic_clean_text(text: object) -> str:
    """Apply deterministic cleaning while preserving sentiment-bearing negation."""
    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"@[\w_]+", " ", value)
    value = re.sub(r"#(?=\w)", "", value)
    value = re.sub(r"(.)\1{2,}", r"\1\1", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"_+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return " ".join(SLANG_MAP.get(token, token) for token in value.split())


@lru_cache(maxsize=100000)
def linguistic_clean_text(text: object) -> str:
    """Add Indonesian stopword handling and stemming to basic cleaning."""
    tokens = [token for token in basic_clean_text(text).split() if token not in STOPWORDS and len(token) > 1]
    return " ".join(
        token if token in NEGATIONS else stem_token(token) for token in tokens
    )


class IndonesianTextPreprocessor(BaseEstimator, TransformerMixin):
    """Serializable raw-text transformer used by training and inference."""
    def __init__(self, variant: str = "basic") -> None:
        self.variant = variant

    def fit(self, X: Iterable[object], y: object = None) -> "IndonesianTextPreprocessor":
        if self.variant not in {"basic", "linguistic"}:
            raise ValueError(f"Unknown preprocessing variant: {self.variant}")
        if self.variant == "linguistic":
            frequencies = Counter(
                token
                for value in X
                for token in basic_clean_text(value).split()
                if token not in STOPWORDS and len(token) > 1
            )
            self.stem_vocabulary_ = {
                token for token, count in frequencies.items() if count >= 5
            }
        return self

    def transform(self, X: Iterable[object]) -> np.ndarray:
        if self.variant == "basic":
            return np.asarray([basic_clean_text(value) for value in X], dtype=object)
        vocabulary = getattr(self, "stem_vocabulary_", set())
        cleaned = []
        for value in X:
            tokens = [
                token for token in basic_clean_text(value).split()
                if token not in STOPWORDS and len(token) > 1
            ]
            cleaned.append(" ".join(
                token if token in NEGATIONS or token not in vocabulary
                else stem_token(token)
                for token in tokens
            ))
        return np.asarray(cleaned, dtype=object)
