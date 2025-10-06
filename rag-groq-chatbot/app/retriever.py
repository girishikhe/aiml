import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple

def tokenize(text: str):
    return [w.lower() for w in text.split() if w.isalpha()]

class SimpleTfidf:
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.fitted = False

    def fit(self, docs: List[str]):
        df = defaultdict(int)
        for doc in docs:
            for w in set(tokenize(doc)):
                df[w] += 1
        N = len(docs)
        self.vocab = {t: i for i, t in enumerate(df.keys())}
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
        self.fitted = True

    def transform(self, docs: List[str]) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Vectorizer not fitted")
        vectors = []
        for doc in docs:
            tokens = tokenize(doc)
            counts = Counter(tokens)
            vec = np.zeros(len(self.vocab))
            for t, c in counts.items():
                if t in self.vocab:
                    vec[self.vocab[t]] = (c / len(tokens)) * self.idf[t]
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)
        return np.vstack(vectors)

class VectorStore:
    def __init__(self):
        self.embeddings = None
        self.texts = []

    def build(self, embeddings: np.ndarray, texts: List[str]):
        self.embeddings = embeddings
        self.texts = texts

    def retrieve(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[str, float]]:
        scores = self.embeddings @ query_vector.T
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.texts[i], float(scores[i])) for i in top_idx]
