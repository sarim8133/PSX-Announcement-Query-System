import json
from psxq.corpus import load_corpus
from psxq.eval.harness import run_trial, run_eval

RECS = load_corpus("data/synthetic/records.json")
GOLD = json.load(open("data/synthetic/gold_questions.json", encoding="utf-8"))

def _fake_pipeline(question, records, today, client=None):
    # deterministic stand-in keyed by question text
    mapping = {
        GOLD[0]["question"]: ("structured", ["S001"], "Alpha Mills 2026-06-25"),
        GOLD[1]["question"]: ("structured", ["S002"], "Beta Cement 7.5 2026-06-30"),
        GOLD[2]["question"]: ("structured", ["S003"], "Gamma Power"),
        GOLD[3]["question"]: ("schema_blocked", [], "not captured"),
        GOLD[4]["question"]: ("schema_blocked", [], "not captured"),
    }
    path, ids, ans = mapping[question]
    return {"path": path, "doc_ids": ids, "answer": ans, "plan": None}

def _fake_judge(question, facts, answer, context, client=None):
    return {"correct": True, "grounded": True}

def test_run_trial_scores_translation_and_schema_coverage(monkeypatch):
    monkeypatch.setattr("psxq.eval.harness.answer_question", _fake_pipeline)
    monkeypatch.setattr("psxq.eval.harness.judge_answer", _fake_judge)
    res = run_trial(GOLD, RECS, today="2026-06-20")
    assert res["translation_exact"] == 1.0          # all doc sets match gold
    assert res["answer_correct"] == 1.0
    assert res["schema_coverage"]["schema_blocked"]["count"] == 2

def test_run_eval_reports_mean_and_range(monkeypatch):
    monkeypatch.setattr("psxq.eval.harness.answer_question", _fake_pipeline)
    monkeypatch.setattr("psxq.eval.harness.judge_answer", _fake_judge)
    out = run_eval(GOLD, RECS, today="2026-06-20", trials=3)
    assert out["translation_exact"]["mean"] == 1.0
    assert out["translation_exact"]["min"] == 1.0 and out["translation_exact"]["max"] == 1.0
    assert out["trials"] == 3
