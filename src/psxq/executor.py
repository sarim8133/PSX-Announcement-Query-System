from __future__ import annotations
from datetime import date
from typing import Any
from psxq.models import Predicate, QueryPlan, Record

def _resolve(record: Record, field: str) -> Any:
    obj: Any = record
    for part in field.split("."):
        if obj is None:
            return None
        obj = obj.get(part) if isinstance(obj, dict) else getattr(obj, part, None)
    return obj

def _as_date(v: Any) -> date | None:
    if v is None:
        return None
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None

def _cmp(value: Any, target: Any):
    """Return (a, b) coerced to dates if both look like dates, else as-is."""
    da, db = _as_date(value), _as_date(target)
    if da is not None and db is not None:
        return da, db
    return value, target

def _match(value: Any, op: str, target: Any) -> bool:
    if op == "eq":
        return value == target
    if op == "neq":
        return value != target
    if op == "in":
        return value in target if isinstance(target, (list, tuple, set)) else False
    if op == "contains":
        if value is None:
            return False
        if isinstance(value, list):
            return target in value
        return str(target).lower() in str(value).lower()
    if op in {"gt", "gte", "lt", "lte"}:
        if value is None or target is None:
            return False
        a, b = _cmp(value, target)
        try:
            if op == "gt":  return a > b
            if op == "gte": return a >= b
            if op == "lt":  return a < b
            if op == "lte": return a <= b
        except TypeError:
            return False
    if op == "between":
        d = _as_date(value)
        lo, hi = _as_date(target[0]), _as_date(target[1])
        return bool(d and lo and hi and lo <= d <= hi)
    if op == "before":
        d, t = _as_date(value), _as_date(target)
        return bool(d and t and d < t)
    if op == "after":
        d, t = _as_date(value), _as_date(target)
        return bool(d and t and d > t)
    raise ValueError(f"unknown op: {op}")

def _matches(record: Record, pred: Predicate) -> bool:
    return _match(_resolve(record, pred.field), pred.op, pred.value)

def execute(plan: QueryPlan, records: list[Record]) -> list[Record]:
    return [r for r in records if all(_matches(r, p) for p in plan.filters)]
