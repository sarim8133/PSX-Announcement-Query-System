from psxq.models import Record, QueryPlan, Predicate

def test_record_defaults_nested_models():
    r = Record(doc_id="S001", listing_date="2026-06-15", company_name="Alpha Mills Limited", subject="x")
    assert r.announcement_signals == []
    assert r.is_actionable_signal is False
    assert r.financials.payout_amount is None

def test_queryplan_parses_predicates_and_path():
    p = QueryPlan(path="structured",
                  filters=[{"field": "financials.payout_amount", "op": "gt", "value": 5}])
    assert p.path == "structured"
    assert isinstance(p.filters[0], Predicate)
    assert p.filters[0].op == "gt"

def test_queryplan_rejects_bad_path():
    import pytest
    with pytest.raises(Exception):
        QueryPlan(path="not_a_path")
