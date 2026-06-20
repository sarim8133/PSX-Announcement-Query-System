from psxq.corpus import load_corpus
from psxq.models import QueryPlan
from psxq.executor import execute

RECS = load_corpus("data/synthetic/records.json")

def ids(records):
    return sorted(r.doc_id for r in records)

def test_contains_on_signal_list():
    plan = QueryPlan(path="structured",
                     filters=[{"field": "announcement_signals", "op": "contains", "value": "Dividend"}])
    assert ids(execute(plan, RECS)) == ["S001", "S002"]

def test_numeric_gt_on_nested_field():
    plan = QueryPlan(path="structured",
                     filters=[{"field": "financials.payout_amount", "op": "gt", "value": 5}])
    assert ids(execute(plan, RECS)) == ["S002"]

def test_date_after_on_meeting_date():
    plan = QueryPlan(path="structured",
                     filters=[{"field": "meeting_details.date", "op": "after", "value": "2026-06-20"}])
    assert ids(execute(plan, RECS)) == ["S001"]

def test_date_window_lte_gte_for_closed_period():
    plan = QueryPlan(path="structured",
                     filters=[{"field": "closure_details.type", "op": "eq", "value": "Closed Period"},
                              {"field": "closure_details.start_date", "op": "lte", "value": "2026-06-20"},
                              {"field": "closure_details.end_date", "op": "gte", "value": "2026-06-20"}])
    assert ids(execute(plan, RECS)) == ["S003"]

def test_predicates_are_anded():
    plan = QueryPlan(path="structured",
                     filters=[{"field": "announcement_signals", "op": "contains", "value": "Dividend"},
                              {"field": "financials.payout_amount", "op": "gt", "value": 5}])
    assert ids(execute(plan, RECS)) == ["S002"]

def test_null_field_does_not_match_numeric():
    plan = QueryPlan(path="structured",
                     filters=[{"field": "financials.payout_amount", "op": "gt", "value": 0}])
    assert ids(execute(plan, RECS)) == ["S002"]

def test_company_substring_contains():
    plan = QueryPlan(path="structured",
                     filters=[{"field": "company_name", "op": "contains", "value": "cement"}])
    assert ids(execute(plan, RECS)) == ["S002"]

from psxq.models import Predicate
from psxq.executor import _match

def test_between_with_malformed_target_returns_false_not_crash():
    # scalar instead of [lo, hi] must fail soft, not raise
    assert _match("2026-06-20", "between", "2026-06-20") is False
    assert _match("2026-06-20", "between", ["2026-06-01"]) is False
    assert _match("2026-06-20", "between", ["2026-06-01", "2026-06-30"]) is True
