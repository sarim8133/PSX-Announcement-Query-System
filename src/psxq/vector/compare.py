from __future__ import annotations
from collections import defaultdict
from psxq.models import Record
from psxq.pipeline import answer_question
from psxq.vector.rag import build_index, rag_answer
from psxq.eval.metrics import doc_set_metrics, recall_at_k, mrr

def compare_arms(gold: list[dict], records: list[Record], today: str,
                 k: int = 4, client=None) -> dict:
    index = build_index(records, client=client)
    spine_exact, path_ok = [], []
    vec_recall, vec_mrr = [], []
    by_cat = defaultdict(lambda: {"spine_exact": [], "vector_recall": []})

    for q in gold:
        cat = q["gold_path"]
        s = answer_question(q["question"], records, today=today, client=client)
        sm = doc_set_metrics(q["gold_doc_ids"], s["doc_ids"])
        spine_exact.append(1.0 if sm["exact"] else 0.0)
        path_ok.append(1.0 if s["path"] == q["gold_path"] else 0.0)
        by_cat[cat]["spine_exact"].append(1.0 if sm["exact"] else 0.0)

        v = rag_answer(q["question"], records, index, k=k, client=client)
        # retrieval metrics only meaningful when gold docs exist (not schema_blocked)
        if q["gold_doc_ids"]:
            vec_recall.append(recall_at_k(v["doc_ids"], q["gold_doc_ids"], k))
            vec_mrr.append(mrr(v["doc_ids"], q["gold_doc_ids"]))
            by_cat[cat]["vector_recall"].append(recall_at_k(v["doc_ids"], q["gold_doc_ids"], k))

    def _mean(xs): return sum(xs) / len(xs) if xs else None
    return {
        "spine": {
            "translation_exact": _mean(spine_exact),
            "path_selection_accuracy": _mean(path_ok),
        },
        "vector": {
            "recall_at_k": _mean(vec_recall),
            "mrr": _mean(vec_mrr),
            "k": k,
        },
        "by_category": {
            cat: {"spine_exact": _mean(v["spine_exact"]),
                  "vector_recall": _mean(v["vector_recall"])}
            for cat, v in by_cat.items()
        },
    }
