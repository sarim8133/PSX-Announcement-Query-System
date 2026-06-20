from __future__ import annotations
from collections import defaultdict
from psxq.models import Record
from psxq.pipeline import answer_question
from psxq.vector.rag import build_index, rag_answer
from psxq.eval.metrics import doc_set_metrics, recall_at_k, mrr

_COMPARISON_NOTE = (
    "spine is set-based (returns an exact predicted set); vector is ranked top-k. "
    "doc_set precision/recall/f1 are computed over the same scored questions and ARE "
    "directly comparable; recall_at_k and mrr are ranking-specific to the vector arm. "
    "schema_blocked questions have no gold docs and are excluded from retrieval scoring."
)

def _mean(xs):
    return sum(xs) / len(xs) if xs else None

def compare_arms(gold: list[dict], records: list[Record], today: str,
                 k: int = 4, client=None) -> dict:
    index = build_index(records, client=client)
    spine_exact, path_ok = [], []
    spine_p, spine_r, spine_f = [], [], []
    vec_recall, vec_mrr = [], []
    vec_p, vec_r, vec_f = [], [], []
    by_cat = defaultdict(lambda: {"spine_exact": [], "vector_recall": [], "vector_mrr": []})

    for q in gold:
        cat = q["gold_path"]
        gold_ids = q["gold_doc_ids"]
        s = answer_question(q["question"], records, today=today, client=client)
        sm = doc_set_metrics(gold_ids, s["doc_ids"])
        spine_exact.append(1.0 if sm["exact"] else 0.0)
        path_ok.append(1.0 if s["path"] == q["gold_path"] else 0.0)
        by_cat[cat]["spine_exact"].append(1.0 if sm["exact"] else 0.0)

        v = rag_answer(q["question"], records, index, k=k, client=client)

        # comparable + ranking metrics only where gold docs exist (not schema_blocked)
        if gold_ids:
            spine_p.append(sm["precision"]); spine_r.append(sm["recall"]); spine_f.append(sm["f1"])
            vm = doc_set_metrics(gold_ids, v["doc_ids"][:k])
            vec_p.append(vm["precision"]); vec_r.append(vm["recall"]); vec_f.append(vm["f1"])
            vec_recall.append(recall_at_k(v["doc_ids"], gold_ids, k))
            vec_mrr.append(mrr(v["doc_ids"], gold_ids))
            by_cat[cat]["vector_recall"].append(recall_at_k(v["doc_ids"], gold_ids, k))
            by_cat[cat]["vector_mrr"].append(mrr(v["doc_ids"], gold_ids))

    return {
        "spine": {
            "translation_exact": _mean(spine_exact),
            "path_selection_accuracy": _mean(path_ok),
            "doc_set": {"precision": _mean(spine_p), "recall": _mean(spine_r), "f1": _mean(spine_f)},
        },
        "vector": {
            "recall_at_k": _mean(vec_recall),
            "mrr": _mean(vec_mrr),
            "k": k,
            "doc_set": {"precision": _mean(vec_p), "recall": _mean(vec_r), "f1": _mean(vec_f)},
        },
        "by_category": {
            cat: {"spine_exact": _mean(v["spine_exact"]),
                  "vector_recall": _mean(v["vector_recall"]),
                  "vector_mrr": _mean(v["vector_mrr"])}
            for cat, v in by_cat.items()
        },
        "note": _COMPARISON_NOTE,
    }
