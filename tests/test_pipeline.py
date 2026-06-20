from psxq.corpus import load_corpus
from psxq.models import QueryPlan
from psxq.pipeline import answer_question

RECS = load_corpus("data/synthetic/records.json")

def _patch(monkeypatch, plan: QueryPlan, answer="ANSWER", semantic_ids=None):
    monkeypatch.setattr("psxq.pipeline.plan_query", lambda q, today, client=None: plan)
    monkeypatch.setattr("psxq.pipeline.generate_answer",
                        lambda q, records, client=None: answer)
    if semantic_ids is not None:
        monkeypatch.setattr("psxq.pipeline.semantic_filter",
                            lambda intent, candidates, client=None:
                                [r for r in candidates if r.doc_id in semantic_ids])

def test_structured_path_returns_filtered_ids(monkeypatch):
    plan = QueryPlan(path="structured",
                     filters=[{"field": "financials.payout_amount", "op": "gt", "value": 5}])
    _patch(monkeypatch, plan)
    res = answer_question("dividends > 5?", RECS, today="2026-06-20")
    assert res["path"] == "structured"
    assert res["doc_ids"] == ["S002"]
    assert res["answer"] == "ANSWER"

def test_schema_blocked_short_circuits_without_records(monkeypatch):
    plan = QueryPlan(path="schema_blocked", schema_blocked_reason="ratio not captured")
    _patch(monkeypatch, plan, answer="Not captured by the schema.")
    res = answer_question("right issue ratio?", RECS, today="2026-06-20")
    assert res["path"] == "schema_blocked"
    assert res["doc_ids"] == []
    assert "Not captured" in res["answer"]

def test_semantic_path_applies_relevance_filter(monkeypatch):
    plan = QueryPlan(path="semantic", semantic_intent="rejected resolutions")
    _patch(monkeypatch, plan, semantic_ids={"S005"})
    res = answer_question("rejected resolutions?", RECS, today="2026-06-20")
    assert res["path"] == "semantic"
    assert res["doc_ids"] == ["S005"]
