from __future__ import annotations
import numpy as np

class VectorIndex:
    """At ~30 docs a 'vector DB' is numpy cosine. Stated honestly (spec §7)."""
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []

    def add(self, doc_id: str, vector) -> None:
        v = np.asarray(vector, dtype=float)
        self._ids.append(doc_id)
        self._vecs.append(v)

    def search(self, query, k: int) -> list[str]:
        if not self._ids:
            return []
        q = np.asarray(query, dtype=float)
        mat = np.vstack(self._vecs)
        qn = q / (np.linalg.norm(q) or 1.0)
        mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        scores = mn @ qn
        order = np.argsort(-scores)[:k]
        return [self._ids[i] for i in order]
