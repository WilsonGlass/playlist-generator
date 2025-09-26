from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class TextEncoder:
    """Thin wrapper around SentenceTransformer to keep imports localized."""

    def __init__(self, model_name: str):
        self._st = SentenceTransformer(model_name)

    def encode_passages(self, lines: List[str]) -> np.ndarray:
        """Encode a list of 'passage:' lines -> [N, D] normalized embeddings."""
        return self._st.encode(lines, normalize_embeddings=True)

    def encode_query(self, q: str) -> np.ndarray:
        """Encode a single 'query:' line -> [1, D] normalized embedding."""
        return self._st.encode([q], normalize_embeddings=True)