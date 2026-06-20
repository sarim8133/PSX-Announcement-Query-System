from __future__ import annotations
from collections import Counter

def doc_set_metrics(gold_ids, pred_ids) -> dict:
    g, p = set(gold_ids), set(pred_ids)
    tp = len(g & p)
    precision = tp / len(p) if p else (1.0 if not g else 0.0)
    recall = tp / len(g) if g else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"exact": g == p, "precision": precision, "recall": recall, "f1": f1}

def recall_at_k(ranked_ids, gold_ids, k: int) -> float:
    g = set(gold_ids)
    if not g:
        return 1.0
    return len(set(ranked_ids[:k]) & g) / len(g)

def hit_at_k(ranked_ids, gold_ids, k: int) -> float:
    g = set(gold_ids)
    return 1.0 if set(ranked_ids[:k]) & g else 0.0

def mrr(ranked_ids, gold_ids) -> float:
    g = set(gold_ids)
    for i, d in enumerate(ranked_ids, start=1):
        if d in g:
            return 1.0 / i
    return 0.0

def schema_coverage(paths) -> dict:
    c = Counter(paths)
    total = len(paths)
    cats = ["structured", "semantic", "schema_blocked"]
    return {k: {"count": c.get(k, 0), "frac": (c.get(k, 0) / total if total else 0.0)}
            for k in cats}

def aggregate(values) -> dict:
    vals = list(values)
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
