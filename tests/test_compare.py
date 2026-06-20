import json
from psxq.corpus import load_corpus
from psxq.vector.compare import compare_arms

RECS = load_corpus("data/synthetic/records.json")
GOLD = json.load(open("data/synthetic/gold_questions.json", encoding="utf-8"))

def test_compare_arms_reports_both_and_path_selection(monkeypatch):
    # spine: perfect structured doc sets + correct path
    def fake_spine(question, records, today, client=None):
        g = next(x for x in GOLD if x["question"] == question)
        return {"path": g["gold_path"], "doc_ids": g["gold_doc_ids"], "answer": "x", "plan": None}
    # vector arm: returns gold docs first (so recall@k = 1 where gold exists)
    def fake_rag(question, records, index, k=4, client=None):
        g = next(x for x in GOLD if x["question"] == question)
        return {"doc_ids": g["gold_doc_ids"] or ["S001"], "answer": "x"}
    monkeypatch.setattr("psxq.vector.compare.answer_question", fake_spine)
    monkeypatch.setattr("psxq.vector.compare.rag_answer", fake_rag)
    monkeypatch.setattr("psxq.vector.compare.build_index", lambda recs, client=None: object())

    out = compare_arms(GOLD, RECS, today="2026-06-20", k=4)
    assert out["spine"]["translation_exact"] == 1.0
    assert out["spine"]["path_selection_accuracy"] == 1.0
    # structured questions have gold docs -> recall measurable; schema_blocked excluded
    assert 0.0 <= out["vector"]["recall_at_k"] <= 1.0
    assert "by_category" in out
