from __future__ import annotations
from psxq.models import Record
from psxq.pipeline import answer_question
from psxq.generator import render_records
from psxq.eval.judge import judge_answer
from psxq.eval.metrics import doc_set_metrics, schema_coverage, aggregate

def run_trial(gold: list[dict], records: list[Record], today: str, client=None) -> dict:
    by_id = {r.doc_id: r for r in records}
    translation_hits, correct_hits, grounded_hits, paths = [], [], [], []
    per_q = []
    for q in gold:
        res = answer_question(q["question"], records, today=today, client=client)
        paths.append(res["path"])
        tm = doc_set_metrics(q["gold_doc_ids"], res["doc_ids"])
        translation_hits.append(1.0 if tm["exact"] else 0.0)
        context = render_records([by_id[d] for d in res["doc_ids"] if d in by_id])
        j = judge_answer(q["question"], q["gold_answer_facts"], res["answer"], context, client=client)
        correct_hits.append(1.0 if j["correct"] else 0.0)
        grounded_hits.append(1.0 if j["grounded"] else 0.0)
        per_q.append({"id": q["id"], "translation": tm, "judge": j, "path": res["path"]})
    n = len(gold)
    return {
        "translation_exact": sum(translation_hits) / n,
        "answer_correct": sum(correct_hits) / n,
        "answer_grounded": sum(grounded_hits) / n,
        "schema_coverage": schema_coverage(paths),
        "per_question": per_q,
    }

def run_eval(gold: list[dict], records: list[Record], today: str,
             trials: int = 3, client=None) -> dict:
    runs = [run_trial(gold, records, today=today, client=client) for _ in range(trials)]
    keys = ["translation_exact", "answer_correct", "answer_grounded"]
    out = {k: aggregate([r[k] for r in runs]) for k in keys}
    out["schema_coverage"] = runs[0]["schema_coverage"]  # deterministic across trials
    out["trials"] = trials
    out["runs"] = runs
    return out
