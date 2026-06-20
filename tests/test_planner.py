import pytest
from psxq.planner import strip_fences, parse_and_validate_plan, plan_query
from psxq.models import QueryPlan

def test_strip_fences_optional_present():
    assert strip_fences('```json\n{"a":1}\n```') == '{"a":1}'

def test_strip_fences_optional_absent():
    assert strip_fences('{"a":1}') == '{"a":1}'

def test_parse_and_validate_plan_ok():
    raw = '{"path":"structured","filters":[{"field":"company_name","op":"contains","value":"Beta"}]}'
    plan = parse_and_validate_plan(raw)
    assert isinstance(plan, QueryPlan)
    assert plan.filters[0].value == "Beta"

def test_parse_and_validate_plan_bad_json_raises():
    with pytest.raises(Exception):
        parse_and_validate_plan("not json")

def test_plan_query_uses_injected_today_and_mocked_model(monkeypatch):
    captured = {}
    def fake_call_model(prompt, client=None):
        captured["prompt"] = prompt
        return '{"path":"structured","filters":[]}'
    monkeypatch.setattr("psxq.planner.call_model", fake_call_model)
    plan = plan_query("anything", today="2026-06-20")
    assert plan.path == "structured"
    assert "2026-06-20" in captured["prompt"]   # today injected
