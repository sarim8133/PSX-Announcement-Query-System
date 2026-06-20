# PSX Announcement Query System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a natural-language query system over PSX announcement extractions that translates questions into structured filters, executes them in Python, and generates grounded answers — measured rigorously — with an optional vector-RAG arm for an honest head-to-head comparison.

**Architecture:** A planner (LLM) turns an NL question into a Pydantic-validated `QueryPlan` (structured predicates + path classification); a mechanical executor applies predicates in Python; a generator writes a grounded answer. An evaluation harness scores translation accuracy and answer correctness over 3 trials (mean + range). Phase 2 adds a fairly-built vector-RAG arm and a comparison harness. The spine (Phase 1) is a complete, shippable project on its own.

**Tech Stack:** Python 3.11+, `google-genai` (Gemini 2.5 Flash), Pydantic v2, numpy (vector arm), pytest. No agent/RAG frameworks — from-scratch is the point.

**Design of record:** `docs/superpowers/specs/2026-06-20-psx-query-system-design.md`

---

## File Structure

```
pyproject.toml            # package + deps
.gitignore                # ignores data/real/, *.pdf, real gold (legal: no data shipped)
README.md                 # methodology + run instructions (data described, not shipped)
src/psxq/
  __init__.py
  models.py               # Record, Predicate, QueryPlan (+ nested detail models)
  corpus.py               # load_corpus(path) -> list[Record]
  llm.py                  # raw Gemini wrapper: call_model(), temp=1.0 + nonce
  planner.py              # NL -> QueryPlan (strip fences, parse, validate by hand)  [THE HARD PART]
  executor.py             # apply QueryPlan predicates over records  [MECHANICAL]
  semantic.py             # lean-A LLM relevance pass for semantic_intent
  generator.py            # records + question -> grounded answer
  pipeline.py             # answer_question(): plan -> execute -> semantic -> generate
  eval/
    __init__.py
    metrics.py            # doc_set_metrics, recall_at_k, hit_at_k, mrr, schema_coverage, aggregate
    judge.py              # LLM judge: answer correctness + faithfulness
    harness.py            # run_eval(): 3 trials, spine metrics, mean+range
  vector/                 # Phase 2 (Path A+)
    __init__.py
    embed.py              # embed_text() via Gemini embeddings
    index.py              # VectorIndex: add / search (numpy cosine)
    rag.py                # pure vector-RAG pipeline
    compare.py            # both arms, same questions, per-category breakdown
data/
  synthetic/
    records.json          # synthetic corpus (publishable, runnable)
    gold_questions.json   # synthetic gold set (publishable, runnable)
  real/                   # gitignored: real PDFs/JSON/gold never committed
tests/
  test_models.py  test_corpus.py  test_executor.py  test_planner.py
  test_semantic.py  test_generator.py  test_pipeline.py
  test_metrics.py  test_judge.py  test_harness.py
  test_vector_index.py  test_rag.py  test_compare.py
```

**Testing strategy:** Deterministic units (models, corpus, executor, metrics, vector index) are tested directly. LLM-touching units (planner, semantic, generator, judge, rag) are tested by **mocking `call_model`/`embed_text`** so tests are offline and deterministic — the from-scratch skill (fence-stripping, JSON parsing, validation, set-compare, recall@k) is exactly what these tests exercise. Live API calls happen only in the eval *runs* (Tasks 12 & 17), gated by `GEMINI_API_KEY`.

---

# Phase 0 — Scaffolding

### Task 0.1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/psxq/__init__.py`, `src/psxq/eval/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "psxq"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["google-genai>=0.3", "pydantic>=2.6", "numpy>=1.26"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`** (legal: no data ever shipped — see spec §3)

```gitignore
__pycache__/
*.pyc
.venv/
.env
data/real/
*.pdf
# real extractions / gold sets are proprietary-compilation data — never commit
data/**/real_*.json
```

- [ ] **Step 3: Create empty package files**

```bash
mkdir -p src/psxq/eval src/psxq/vector data/synthetic data/real tests
touch src/psxq/__init__.py src/psxq/eval/__init__.py src/psxq/vector/__init__.py tests/__init__.py
echo "*" > data/real/.gitignore   # keep dir, ignore contents
```

- [ ] **Step 4: Create `README.md` stub**

```markdown
# PSX Announcement Query System

NL questions over PSX corporate-announcement extractions: question -> structured filter -> grounded answer, measured. Optional vector-RAG arm for an honest comparison. See `docs/superpowers/specs/`.

**Data is not shipped** (PSX prohibits redistribution). The repo ships synthetic examples in `data/synthetic/`; real PDFs/JSON stay local and gitignored. Scraping PSX is prohibited — source PDFs are downloaded manually.

## Run
`pip install -e ".[dev]"` then `pytest`. To run eval on real data: set `GEMINI_API_KEY`, point the harness at `data/real/`.
```

- [ ] **Step 5: Install and verify, then commit**

Run: `pip install -e ".[dev]" && pytest -q`
Expected: pytest runs, collects 0 tests, exits 0.

```bash
git init
git add pyproject.toml .gitignore README.md src tests data
git commit -m "chore: scaffold psxq project"
```

---

### Task 0.2: Synthetic corpus and gold set

**Files:**
- Create: `data/synthetic/records.json`, `data/synthetic/gold_questions.json`

- [ ] **Step 1: Create `data/synthetic/records.json`** (5 synthetic records covering the path categories; no real PSX data)

```json
[
  {"doc_id": "S001", "listing_date": "2026-06-15", "document_date": "2026-06-15",
   "company_name": "Alpha Mills Limited", "subject": "Board Meeting to consider dividend",
   "announcement_signals": ["Board Meeting", "Dividend"], "is_actionable_signal": true,
   "meeting_details": {"date": "2026-06-25", "time": "11:00 AM"},
   "closure_details": {"start_date": null, "end_date": null, "type": null},
   "financials": {"payout_amount": null, "currency": null, "payment_due_date": null, "entitlement_record_date": null},
   "summary": "Alpha Mills board meets 2026-06-25 to consider a cash dividend."},
  {"doc_id": "S002", "listing_date": "2026-06-10", "document_date": "2026-06-09",
   "company_name": "Beta Cement Limited", "subject": "Cash dividend declaration",
   "announcement_signals": ["Dividend", "Profit Payment"], "is_actionable_signal": true,
   "meeting_details": {"date": null, "time": null},
   "closure_details": {"start_date": null, "end_date": null, "type": null},
   "financials": {"payout_amount": 7.5, "currency": "PKR", "payment_due_date": "2026-07-05", "entitlement_record_date": "2026-06-30"},
   "summary": "Beta Cement declares PKR 7.5/share dividend, record date 2026-06-30."},
  {"doc_id": "S003", "listing_date": "2026-06-12", "document_date": "2026-06-12",
   "company_name": "Gamma Power Limited", "subject": "Closed period notice",
   "announcement_signals": ["Closed Period"], "is_actionable_signal": true,
   "meeting_details": {"date": null, "time": null},
   "closure_details": {"start_date": "2026-06-16", "end_date": "2026-06-30", "type": "Closed Period"},
   "financials": {"payout_amount": null, "currency": null, "payment_due_date": null, "entitlement_record_date": null},
   "summary": "Gamma Power closed period 2026-06-16 to 2026-06-30."},
  {"doc_id": "S004", "listing_date": "2026-06-08", "document_date": "2026-06-08",
   "company_name": "Delta Textiles Limited", "subject": "Right Issue announcement",
   "announcement_signals": ["Right Issue"], "is_actionable_signal": true,
   "meeting_details": {"date": null, "time": null},
   "closure_details": {"start_date": null, "end_date": null, "type": null},
   "financials": {"payout_amount": null, "currency": null, "payment_due_date": null, "entitlement_record_date": null},
   "summary": "Delta Textiles announces a right issue (ratio/price not captured by schema)."},
  {"doc_id": "S005", "listing_date": "2026-06-05", "document_date": "2026-06-05",
   "company_name": "Epsilon Foods Limited", "subject": "AGM resolutions",
   "announcement_signals": ["General Meeting"], "is_actionable_signal": false,
   "meeting_details": {"date": "2026-06-05", "time": "10:00 AM"},
   "closure_details": {"start_date": null, "end_date": null, "type": null},
   "financials": {"payout_amount": null, "currency": null, "payment_due_date": null, "entitlement_record_date": null},
   "summary": "Epsilon Foods held AGM; some resolutions outcome (not captured by schema)."}
]
```

- [ ] **Step 2: Create `data/synthetic/gold_questions.json`** (covers structured, semantic, schema_blocked)

```json
[
  {"id": "q1", "question": "Which companies have a board meeting after 2026-06-20 to consider a dividend?",
   "gold_path": "structured",
   "gold_filters": [{"field": "announcement_signals", "op": "contains", "value": "Board Meeting"},
                    {"field": "meeting_details.date", "op": "after", "value": "2026-06-20"}],
   "gold_doc_ids": ["S001"],
   "gold_answer_facts": ["Alpha Mills Limited", "2026-06-25"]},
  {"id": "q2", "question": "List dividends with payout above PKR 5 per share and their record dates.",
   "gold_path": "structured",
   "gold_filters": [{"field": "announcement_signals", "op": "contains", "value": "Dividend"},
                    {"field": "financials.payout_amount", "op": "gt", "value": 5}],
   "gold_doc_ids": ["S002"],
   "gold_answer_facts": ["Beta Cement Limited", "7.5", "2026-06-30"]},
  {"id": "q3", "question": "Which companies are in a closed period covering 2026-06-20?",
   "gold_path": "structured",
   "gold_filters": [{"field": "closure_details.type", "op": "eq", "value": "Closed Period"},
                    {"field": "closure_details.start_date", "op": "lte", "value": "2026-06-20"},
                    {"field": "closure_details.end_date", "op": "gte", "value": "2026-06-20"}],
   "gold_doc_ids": ["S003"],
   "gold_answer_facts": ["Gamma Power Limited"]},
  {"id": "q4", "question": "What is the subscription price and ratio of any recent right issue?",
   "gold_path": "schema_blocked",
   "gold_filters": [],
   "gold_doc_ids": [],
   "gold_answer_facts": ["not captured"]},
  {"id": "q5", "question": "Did any general meeting reject or fail a resolution?",
   "gold_path": "schema_blocked",
   "gold_filters": [],
   "gold_doc_ids": [],
   "gold_answer_facts": ["not captured"]}
]
```

- [ ] **Step 3: Verify JSON parses**

Run: `python -c "import json; json.load(open('data/synthetic/records.json')); json.load(open('data/synthetic/gold_questions.json')); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add data/synthetic
git commit -m "test: add synthetic corpus and gold questions"
```

---

# Phase 1 — The Spine (standalone shippable system)

### Task 1: Data models

**Files:**
- Create: `src/psxq/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.models'`

- [ ] **Step 3: Write `src/psxq/models.py`**

```python
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

class MeetingDetails(BaseModel):
    date: Optional[str] = None
    time: Optional[str] = None

class ClosureDetails(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    type: Optional[str] = None  # "Closed Period" | "Book Closure" | None

class Financials(BaseModel):
    payout_amount: Optional[float] = None
    currency: Optional[str] = None
    payment_due_date: Optional[str] = None
    entitlement_record_date: Optional[str] = None

class Record(BaseModel):
    doc_id: str
    listing_date: str
    document_date: Optional[str] = None
    company_name: str
    subject: str
    announcement_signals: list[str] = Field(default_factory=list)
    is_actionable_signal: bool = False
    meeting_details: MeetingDetails = Field(default_factory=MeetingDetails)
    closure_details: ClosureDetails = Field(default_factory=ClosureDetails)
    financials: Financials = Field(default_factory=Financials)
    summary: str = ""

PathType = Literal["structured", "semantic", "schema_blocked"]
OpType = Literal["eq", "neq", "in", "contains", "gt", "gte", "lt", "lte", "between", "before", "after"]

class Predicate(BaseModel):
    field: str
    op: OpType
    value: Any = None

class QueryPlan(BaseModel):
    path: PathType
    filters: list[Predicate] = Field(default_factory=list)
    semantic_intent: Optional[str] = None
    schema_blocked_reason: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/psxq/models.py tests/test_models.py
git commit -m "feat: add Record and QueryPlan models"
```

---

### Task 2: Corpus loader

**Files:**
- Create: `src/psxq/corpus.py`
- Test: `tests/test_corpus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus.py
from psxq.corpus import load_corpus
from psxq.models import Record

def test_load_corpus_returns_records():
    recs = load_corpus("data/synthetic/records.json")
    assert len(recs) == 5
    assert all(isinstance(r, Record) for r in recs)
    assert recs[0].doc_id == "S001"
    assert recs[1].financials.payout_amount == 7.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.corpus'`

- [ ] **Step 3: Write `src/psxq/corpus.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
from psxq.models import Record

def load_corpus(path: str | Path) -> list[Record]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Record(**r) for r in raw]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_corpus.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psxq/corpus.py tests/test_corpus.py
git commit -m "feat: add corpus loader"
```

---

### Task 3: Executor (mechanical predicate evaluation)

**Files:**
- Create: `src/psxq/executor.py`
- Test: `tests/test_executor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_executor.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.executor'`

- [ ] **Step 3: Write `src/psxq/executor.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_executor.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/psxq/executor.py tests/test_executor.py
git commit -m "feat: add mechanical predicate executor"
```

---

### Task 4: Raw LLM client

**Files:**
- Create: `src/psxq/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test** (tests the nonce/temperature wiring against a fake client, no network)

```python
# tests/test_llm.py
from psxq.llm import call_model

class _FakeResp:
    text = '{"ok": true}'

class _FakeModels:
    def __init__(self): self.last = None
    def generate_content(self, **kwargs):
        self.last = kwargs
        return _FakeResp()

class _FakeClient:
    def __init__(self): self.models = _FakeModels()

def test_call_model_passes_temperature_and_returns_text():
    client = _FakeClient()
    out = call_model("hello", client=client)
    assert out == '{"ok": true}'
    assert client.models.last["config"]["temperature"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.llm'`

- [ ] **Step 3: Write `src/psxq/llm.py`**

```python
from __future__ import annotations
import os

_DEFAULT_MODEL = "gemini-2.5-flash"

def get_client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def call_model(prompt: str, client=None, model: str = _DEFAULT_MODEL,
               temperature: float = 1.0) -> str:
    client = client or get_client()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": temperature},
    )
    return resp.text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psxq/llm.py tests/test_llm.py
git commit -m "feat: add raw Gemini client wrapper (temp 1.0)"
```

---

### Task 5: Planner — NL → QueryPlan (THE HARD PART)

**Files:**
- Create: `src/psxq/planner.py`
- Test: `tests/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_planner.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.planner'`

- [ ] **Step 3: Write `src/psxq/planner.py`**

```python
from __future__ import annotations
import json
import re
import time
from psxq.llm import call_model
from psxq.models import QueryPlan

def strip_fences(raw: str) -> str:
    """Remove OPTIONAL markdown fences. The model is told not to use them,
    so fences must be treated as optional, never required (P1 recurring bug)."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_and_validate_plan(raw: str) -> QueryPlan:
    data = json.loads(strip_fences(raw))
    return QueryPlan(**data)

PLANNER_PROMPT = """You translate a question about PSX corporate announcements into a JSON query plan.
Today's date is {today}. Resolve relative dates (e.g. "next 2 weeks") to absolute YYYY-MM-DD.

Record fields you may filter on:
- company_name (str), subject (str), listing_date, document_date (YYYY-MM-DD)
- announcement_signals (list of: Board Meeting, Closed Period, Book Closure, Dividend,
  Bonus Issue, Right Issue, General Meeting, Director Election, Profit Payment, Other)
- is_actionable_signal (bool)
- meeting_details.date, meeting_details.time
- closure_details.start_date, closure_details.end_date, closure_details.type
- financials.payout_amount (number), financials.currency,
  financials.payment_due_date, financials.entitlement_record_date

Operators: eq, neq, in, contains, gt, gte, lt, lte, between, before, after.
For list fields (announcement_signals) use "contains" with a single value.

Choose exactly one path:
- "structured": answerable by filtering the fields above.
- "semantic": genuinely fuzzy intent not expressible as field predicates.
- "schema_blocked": the answer is NOT representable in the schema (right/bonus issue
  ratio or subscription price, multi-tranche record dates, meeting outcomes, Shariah
  compliance). Set schema_blocked_reason. Do NOT invent fields.

Output ONLY JSON, no markdown fences:
{{"path": "...", "filters": [{{"field": "...", "op": "...", "value": ...}}],
  "semantic_intent": null, "schema_blocked_reason": null}}

Question: {question}
[nonce:{nonce}]"""

def plan_query(question: str, today: str, client=None) -> QueryPlan:
    prompt = PLANNER_PROMPT.format(question=question, today=today, nonce=time.time_ns())
    raw = call_model(prompt, client=client)
    return parse_and_validate_plan(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_planner.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/psxq/planner.py tests/test_planner.py
git commit -m "feat: add NL->QueryPlan planner with optional-fence stripping"
```

---

### Task 6: Semantic relevance pass (lean A)

**Files:**
- Create: `src/psxq/semantic.py`
- Test: `tests/test_semantic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_semantic.py
from psxq.corpus import load_corpus
from psxq.semantic import semantic_filter

RECS = load_corpus("data/synthetic/records.json")

def test_semantic_filter_keeps_ids_the_model_returns(monkeypatch):
    monkeypatch.setattr("psxq.semantic.call_model", lambda prompt, client=None: '["S005"]')
    out = semantic_filter("meetings where something was rejected", RECS)
    assert [r.doc_id for r in out] == ["S005"]

def test_semantic_filter_ignores_unknown_ids(monkeypatch):
    monkeypatch.setattr("psxq.semantic.call_model", lambda prompt, client=None: '["NOPE","S002"]')
    out = semantic_filter("x", RECS)
    assert [r.doc_id for r in out] == ["S002"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_semantic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.semantic'`

- [ ] **Step 3: Write `src/psxq/semantic.py`**

```python
from __future__ import annotations
import json
import time
from psxq.llm import call_model
from psxq.models import Record
from psxq.planner import strip_fences

SEMANTIC_PROMPT = """Given a fuzzy information need and candidate announcements, return a JSON
array of the doc_ids that genuinely satisfy the need. Return only doc_ids present below.

Need: {intent}

Candidates:
{candidates}

Output ONLY a JSON array of doc_id strings, no markdown.
[nonce:{nonce}]"""

def _render(records: list[Record]) -> str:
    return "\n".join(f'{r.doc_id}: {r.subject} | {r.summary}' for r in records)

def semantic_filter(intent: str, candidates: list[Record], client=None) -> list[Record]:
    prompt = SEMANTIC_PROMPT.format(intent=intent, candidates=_render(candidates),
                                    nonce=time.time_ns())
    raw = call_model(prompt, client=client)
    keep = set(json.loads(strip_fences(raw)))
    return [r for r in candidates if r.doc_id in keep]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_semantic.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psxq/semantic.py tests/test_semantic.py
git commit -m "feat: add lean-A semantic relevance pass"
```

---

### Task 7: Answer generator (grounded)

**Files:**
- Create: `src/psxq/generator.py`
- Test: `tests/test_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator.py
from psxq.corpus import load_corpus
from psxq.generator import render_records, generate_answer

RECS = load_corpus("data/synthetic/records.json")

def test_render_records_includes_company_and_facts():
    text = render_records([RECS[1]])
    assert "Beta Cement Limited" in text
    assert "7.5" in text

def test_generate_answer_passes_records_into_prompt(monkeypatch):
    captured = {}
    def fake(prompt, client=None):
        captured["prompt"] = prompt
        return "Beta Cement Limited pays PKR 7.5/share."
    monkeypatch.setattr("psxq.generator.call_model", fake)
    out = generate_answer("dividends above 5?", [RECS[1]])
    assert "Beta Cement" in out
    assert "Beta Cement Limited" in captured["prompt"]

def test_generate_answer_empty_records_still_calls_model(monkeypatch):
    monkeypatch.setattr("psxq.generator.call_model",
                        lambda prompt, client=None: "No matching announcements found.")
    out = generate_answer("anything?", [])
    assert "No matching" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.generator'`

- [ ] **Step 3: Write `src/psxq/generator.py`**

```python
from __future__ import annotations
import time
from psxq.llm import call_model
from psxq.models import Record

GEN_PROMPT = """Answer the question using ONLY the announcement records below.
Every claim must be traceable to a record. If the records do not contain the answer,
say so plainly — never extrapolate or invent values.

Question: {question}

Records:
{records}

Answer concisely.
[nonce:{nonce}]"""

def render_records(records: list[Record]) -> str:
    if not records:
        return "(no matching records)"
    lines = []
    for r in records:
        lines.append(
            f"- {r.doc_id} | {r.company_name} | {r.subject} | "
            f"signals={r.announcement_signals} | actionable={r.is_actionable_signal} | "
            f"meeting={r.meeting_details.date} {r.meeting_details.time} | "
            f"closure={r.closure_details.start_date}..{r.closure_details.end_date} ({r.closure_details.type}) | "
            f"payout={r.financials.payout_amount} {r.financials.currency} "
            f"record={r.financials.entitlement_record_date}"
        )
    return "\n".join(lines)

def generate_answer(question: str, records: list[Record], client=None) -> str:
    prompt = GEN_PROMPT.format(question=question, records=render_records(records),
                               nonce=time.time_ns())
    return call_model(prompt, client=client)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generator.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/psxq/generator.py tests/test_generator.py
git commit -m "feat: add grounded answer generator"
```

---

### Task 8: Pipeline wiring

**Files:**
- Create: `src/psxq/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.pipeline'`

- [ ] **Step 3: Write `src/psxq/pipeline.py`**

```python
from __future__ import annotations
from psxq.models import Record
from psxq.planner import plan_query
from psxq.executor import execute
from psxq.semantic import semantic_filter
from psxq.generator import generate_answer

def answer_question(question: str, records: list[Record], today: str, client=None) -> dict:
    plan = plan_query(question, today=today, client=client)

    if plan.path == "schema_blocked":
        selected: list[Record] = []
    elif plan.path == "semantic":
        candidates = execute(plan, records) if plan.filters else records
        selected = semantic_filter(plan.semantic_intent or question, candidates, client=client)
    else:  # structured
        selected = execute(plan, records)

    answer = generate_answer(question, selected, client=client)
    return {
        "path": plan.path,
        "doc_ids": [r.doc_id for r in selected],
        "answer": answer,
        "plan": plan,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/psxq/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire planner->executor->semantic->generator pipeline"
```

---

### Task 9: Metrics

**Files:**
- Create: `src/psxq/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from psxq.eval.metrics import (doc_set_metrics, recall_at_k, hit_at_k, mrr,
                               schema_coverage, aggregate)

def test_doc_set_metrics_exact_and_prf():
    m = doc_set_metrics(["A", "B"], ["A", "B"])
    assert m["exact"] is True and m["precision"] == 1.0 and m["recall"] == 1.0
    m2 = doc_set_metrics(["A", "B"], ["A", "C"])
    assert m2["exact"] is False and m2["precision"] == 0.5 and m2["recall"] == 0.5

def test_doc_set_metrics_empty_gold_and_pred_is_exact():
    m = doc_set_metrics([], [])
    assert m["exact"] is True

def test_recall_and_hit_at_k():
    assert recall_at_k(["A", "B", "C"], ["B"], k=2) == 1.0
    assert hit_at_k(["A", "B", "C"], ["Z"], k=2) == 0.0

def test_mrr_rank_position():
    assert mrr(["A", "B", "C"], ["B"]) == 0.5
    assert mrr(["A", "B"], ["Z"]) == 0.0

def test_schema_coverage_counts_fractions():
    cov = schema_coverage(["structured", "structured", "schema_blocked", "semantic"])
    assert cov["structured"]["count"] == 2
    assert cov["schema_blocked"]["frac"] == 0.25

def test_aggregate_mean_min_max():
    a = aggregate([2.0, 4.0, 3.0])
    assert a["mean"] == 3.0 and a["min"] == 2.0 and a["max"] == 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.eval.metrics'`

- [ ] **Step 3: Write `src/psxq/eval/metrics.py`**

```python
from __future__ import annotations
from collections import Counter

def doc_set_metrics(gold_ids, pred_ids) -> dict:
    g, p = set(gold_ids), set(pred_ids)
    tp = len(g & p)
    precision = tp / len(p) if p else (1.0 if not g else 0.0)
    recall = tp / len(g) if g else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"exact": g == p, "precision": precision, "recall": recall, "f1": f1}

def recall_at_k(ranked_ids, gold_ids, k: int) -> float:
    g = set(gold_ids)
    if not g:
        return 1.0
    return len(set(ranked_ids[:k]) & g) / len(g)

def hit_at_k(ranked_ids, gold_ids, k: int) -> float:
    g = set(gold_ids)
    return 1.0 if set(ranked_ids[:k]) & g else 0.0

def mrr(ranked_ids, gold_ids) -> float:
    g = set(gold_ids)
    for i, d in enumerate(ranked_ids, start=1):
        if d in g:
            return 1.0 / i
    return 0.0

def schema_coverage(paths) -> dict:
    c = Counter(paths)
    total = len(paths)
    cats = ["structured", "semantic", "schema_blocked"]
    return {k: {"count": c.get(k, 0), "frac": (c.get(k, 0) / total if total else 0.0)}
            for k in cats}

def aggregate(values) -> dict:
    vals = list(values)
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/psxq/eval/metrics.py tests/test_metrics.py
git commit -m "feat: add evaluation metrics"
```

---

### Task 10: LLM judge (answer correctness + faithfulness)

**Files:**
- Create: `src/psxq/eval/judge.py`
- Test: `tests/test_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_judge.py
from psxq.eval.judge import judge_answer

def test_judge_parses_correct_and_grounded(monkeypatch):
    monkeypatch.setattr("psxq.eval.judge.call_model",
                        lambda prompt, client=None: '{"correct": true, "grounded": true}')
    out = judge_answer("q", ["Beta Cement Limited", "7.5"], "Beta Cement pays 7.5", "context")
    assert out == {"correct": True, "grounded": True}

def test_judge_handles_fenced_json(monkeypatch):
    monkeypatch.setattr("psxq.eval.judge.call_model",
                        lambda prompt, client=None: '```json\n{"correct": false, "grounded": true}\n```')
    out = judge_answer("q", ["x"], "wrong", "ctx")
    assert out["correct"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.eval.judge'`

- [ ] **Step 3: Write `src/psxq/eval/judge.py`**

```python
from __future__ import annotations
import json
import time
from psxq.llm import call_model
from psxq.planner import strip_fences

JUDGE_PROMPT = """You are scoring an answer about PSX announcements.
Question: {question}
Required facts (gold): {facts}
Retrieved context the answer was allowed to use:
{context}
Answer to score:
{answer}

Return JSON only:
{{"correct": <true if the answer states all required gold facts and contradicts none>,
  "grounded": <true if every claim in the answer is supported by the retrieved context>}}
[nonce:{nonce}]"""

def judge_answer(question: str, gold_facts, answer: str, context: str, client=None) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, facts=gold_facts, context=context,
                                 answer=answer, nonce=time.time_ns())
    raw = call_model(prompt, client=client)
    data = json.loads(strip_fences(raw))
    return {"correct": bool(data["correct"]), "grounded": bool(data["grounded"])}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_judge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psxq/eval/judge.py tests/test_judge.py
git commit -m "feat: add LLM judge for answer correctness and faithfulness"
```

---

### Task 11: Evaluation harness (3 trials, mean + range)

**Files:**
- Create: `src/psxq/eval/harness.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.eval.harness'`

- [ ] **Step 3: Write `src/psxq/eval/harness.py`**

```python
from __future__ import annotations
from psxq.models import Record
from psxq.pipeline import answer_question
from psxq.generator import render_records
from psxq.executor import execute
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_harness.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/psxq/eval/harness.py tests/test_harness.py
git commit -m "feat: add eval harness with 3-trial mean+range"
```

---

### Task 12: Spine milestone — run on real data + write-up section

**Files:**
- Create: `src/psxq/run_eval.py` (CLI entry)
- Modify: `README.md` (add results section)

- [ ] **Step 1: Write the CLI entry `src/psxq/run_eval.py`**

```python
from __future__ import annotations
import argparse
import json
from psxq.corpus import load_corpus
from psxq.eval.harness import run_eval

def main() -> None:
    ap = argparse.ArgumentParser(description="Run PSX query-system evaluation.")
    ap.add_argument("--records", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--today", required=True, help="YYYY-MM-DD, anchors relative dates")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    records = load_corpus(args.records)
    gold = json.loads(open(args.gold, encoding="utf-8").read())
    out = run_eval(gold, records, today=args.today, trials=args.trials)
    summary = {k: out[k] for k in ["translation_exact", "answer_correct", "answer_grounded"]}
    print(json.dumps({"summary": summary, "schema_coverage": out["schema_coverage"],
                      "trials": out["trials"]}, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on synthetic data (offline-safe wiring check)**

This run hits the real API only if `GEMINI_API_KEY` is set. With no key it should fail fast at the client, confirming wiring. To validate the CLI itself offline, run the full test suite instead:

Run: `pytest -q`
Expected: all tests pass (entire spine green).

- [ ] **Step 3: (Author, needs `GEMINI_API_KEY`) Build the real corpus and gold set**

Place real extractions at `data/real/records.json` and the hand-verified gold at `data/real/gold_questions.json` (both gitignored). The author writes/verifies gold values — this is the skill being built (PROJECT_CONTEXT §8.5).

Two separate efforts, sized differently:
- **Corpus = per-document extraction (no labeling).** Run P1's extractor over every PDF, including newly added ones. Extra docs are welcome — they act as distractors that make retrieval/filtering non-trivial. Corpus size is not capped at 30.
- **Gold = per-question, ~30 questions total** (NOT one per doc), each labeled with correct path, relevant `doc_ids`, and gold answer facts.

**Gold re-verification on corpus change (P1 "gold integrity is sacred"):** whenever documents are added to the corpus, re-check every existing question whose relevant-doc set or answer facts a new doc could affect (e.g. a new >5 dividend changes a "dividends above PKR 5" answer set) and update its gold. Add new *questions* only to cover categories not yet exercised (e.g. first Bonus Issue example).

- [ ] **Step 4: (Author) Run the real evaluation, 3 trials**

Run: `python -m psxq.run_eval --records data/real/records.json --gold data/real/gold_questions.json --today 2026-06-20 --trials 3`
Expected: JSON with `translation_exact`, `answer_correct`, `answer_grounded` (mean/min/max) and `schema_coverage`.

- [ ] **Step 5: Add results to `README.md` and commit**

Record the two spine metrics with **mean AND range** (never single-run), plus the schema-coverage finding ("X% of questions are schema-blocked"). Note the planner is where errors concentrate.

```bash
git add src/psxq/run_eval.py README.md
git commit -m "feat: add eval CLI and spine results (mean+range)"
```

> **This is the shippable milestone.** The spine is a complete project measured on two metrics. If Project 3's timeline is threatened, stop here and label Phase 2 "future work" (spec §8.5).

---

# Phase 2 — Vector-RAG Arm + Comparison (Path A+, gated on the job-posting checkbox, spec §2)

### Task 13: Embedding function

**Files:**
- Create: `src/psxq/vector/embed.py`
- Test: `tests/test_embed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed.py
from psxq.vector.embed import embed_text

def test_embed_text_returns_vector(monkeypatch):
    class _R:  # shape mirrors google-genai embed response
        embeddings = [type("E", (), {"values": [0.1, 0.2, 0.3]})()]
    class _M:
        def embed_content(self, **kwargs): return _R()
    class _C:
        models = _M()
    vec = embed_text("hello", client=_C())
    assert vec == [0.1, 0.2, 0.3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.vector.embed'`

- [ ] **Step 3: Write `src/psxq/vector/embed.py`**

```python
from __future__ import annotations
from psxq.llm import get_client

_EMBED_MODEL = "gemini-embedding-001"

def embed_text(text: str, client=None, model: str = _EMBED_MODEL) -> list[float]:
    client = client or get_client()
    resp = client.models.embed_content(model=model, contents=text)
    return list(resp.embeddings[0].values)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psxq/vector/embed.py tests/test_embed.py
git commit -m "feat: add embedding function"
```

---

### Task 14: Vector index (numpy cosine)

**Files:**
- Create: `src/psxq/vector/index.py`
- Test: `tests/test_vector_index.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_index.py
from psxq.vector.index import VectorIndex

def test_search_ranks_by_cosine():
    idx = VectorIndex()
    idx.add("A", [1.0, 0.0])
    idx.add("B", [0.0, 1.0])
    idx.add("C", [0.9, 0.1])
    ranked = idx.search([1.0, 0.0], k=2)
    assert ranked == ["A", "C"]

def test_search_k_limits_results():
    idx = VectorIndex()
    for i, v in enumerate([[1, 0], [0.8, 0.2], [0, 1]]):
        idx.add(f"D{i}", v)
    assert len(idx.search([1, 0], k=1)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vector_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.vector.index'`

- [ ] **Step 3: Write `src/psxq/vector/index.py`**

```python
from __future__ import annotations
import numpy as np

class VectorIndex:
    """At ~30 docs a 'vector DB' is numpy cosine. Stated honestly (spec §7)."""
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []

    def add(self, doc_id: str, vector) -> None:
        v = np.asarray(vector, dtype=float)
        self._ids.append(doc_id)
        self._vecs.append(v)

    def search(self, query, k: int) -> list[str]:
        if not self._ids:
            return []
        q = np.asarray(query, dtype=float)
        mat = np.vstack(self._vecs)
        qn = q / (np.linalg.norm(q) or 1.0)
        mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        scores = mn @ qn
        order = np.argsort(-scores)[:k]
        return [self._ids[i] for i in order]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vector_index.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psxq/vector/index.py tests/test_vector_index.py
git commit -m "feat: add numpy cosine vector index"
```

---

### Task 15: Pure vector-RAG pipeline

**Files:**
- Create: `src/psxq/vector/rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag.py
from psxq.corpus import load_corpus
from psxq.vector.rag import build_index, rag_answer

RECS = load_corpus("data/synthetic/records.json")

def _fake_embed(text, client=None):
    # toy embedding: presence of keywords -> 3-dim vector
    t = text.lower()
    return [float("dividend" in t), float("closed period" in t or "closure" in t),
            float("right issue" in t)]

def test_build_index_embeds_each_record(monkeypatch):
    monkeypatch.setattr("psxq.vector.rag.embed_text", _fake_embed)
    idx = build_index(RECS)
    assert set(idx._ids) == {"S001", "S002", "S003", "S004", "S005"}

def test_rag_answer_retrieves_then_generates(monkeypatch):
    monkeypatch.setattr("psxq.vector.rag.embed_text", _fake_embed)
    monkeypatch.setattr("psxq.vector.rag.generate_answer",
                        lambda q, records, client=None: "GEN:" + ",".join(r.doc_id for r in records))
    idx = build_index(RECS)
    res = rag_answer("any dividend news?", RECS, idx, k=2)
    assert "S002" in res["doc_ids"]      # dividend record retrieved
    assert res["answer"].startswith("GEN:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.vector.rag'`

- [ ] **Step 3: Write `src/psxq/vector/rag.py`**

```python
from __future__ import annotations
from psxq.models import Record
from psxq.generator import generate_answer
from psxq.vector.embed import embed_text
from psxq.vector.index import VectorIndex

def _doc_text(r: Record) -> str:
    return f"{r.company_name}. {r.subject}. signals: {', '.join(r.announcement_signals)}. {r.summary}"

def build_index(records: list[Record], client=None) -> VectorIndex:
    idx = VectorIndex()
    for r in records:
        idx.add(r.doc_id, embed_text(_doc_text(r), client=client))
    return idx

def rag_answer(question: str, records: list[Record], index: VectorIndex,
               k: int = 4, client=None) -> dict:
    by_id = {r.doc_id: r for r in records}
    qvec = embed_text(question, client=client)
    ranked = index.search(qvec, k=k)
    selected = [by_id[d] for d in ranked if d in by_id]
    answer = generate_answer(question, selected, client=client)
    return {"doc_ids": ranked, "answer": answer}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psxq/vector/rag.py tests/test_rag.py
git commit -m "feat: add pure vector-RAG pipeline"
```

---

### Task 16: Comparison harness (both arms, per-category)

**Files:**
- Create: `src/psxq/vector/compare.py`
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psxq.vector.compare'`

- [ ] **Step 3: Write `src/psxq/vector/compare.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compare.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psxq/vector/compare.py tests/test_compare.py
git commit -m "feat: add spine-vs-vector comparison harness"
```

---

### Task 17: Comparison run + write-up

**Files:**
- Create: `src/psxq/run_compare.py`
- Modify: `README.md`

- [ ] **Step 1: Write `src/psxq/run_compare.py`**

```python
from __future__ import annotations
import argparse
import json
from psxq.corpus import load_corpus
from psxq.vector.compare import compare_arms

def main() -> None:
    ap = argparse.ArgumentParser(description="Compare structured spine vs vector-RAG.")
    ap.add_argument("--records", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--today", required=True)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    records = load_corpus(args.records)
    gold = json.loads(open(args.gold, encoding="utf-8").read())
    print(json.dumps(compare_arms(gold, records, today=args.today, k=args.k), indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify suite green**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 3: (Author, needs `GEMINI_API_KEY`) Run the comparison on real data**

Run: `python -m psxq.run_compare --records data/real/records.json --gold data/real/gold_questions.json --today 2026-06-20 --k 4`
Expected: JSON with `spine`, `vector`, and `by_category` breakdown.

- [ ] **Step 4: Write the comparison up in `README.md`**

Report, per category: spine translation-exact vs vector recall@k. State the hypothesis and whether it held — structured wins on filters, vector ties/wins on the fuzzy minority. Keep the honesty guardrails (don't oversell a 30-vector cosine as a "vector DB"; the skill is the measured comparison and the judgment).

- [ ] **Step 5: Commit**

```bash
git add src/psxq/run_compare.py README.md
git commit -m "feat: add comparison CLI and write-up"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** structured spine (Tasks 1–8) ✓; planner-is-hard-part framing (Task 5) ✓; mechanical executor (Task 3) ✓; three path categories incl. schema_blocked (models Task 1, planner Task 5, pipeline Task 8) ✓; schema-coverage finding (metrics Task 9, harness Task 11) ✓; spine ships on 2 metrics — translation + answer correctness/faithfulness (Tasks 9–12) ✓; arm adds retrieval recall@k + path-selection (Tasks 13–16) ✓; vector arm built fairly + per-category comparison (Tasks 15–16) ✓; 3 trials mean+range (Task 11) ✓; temp 1.0 + nonce (Task 4, used everywhere) ✓; from-scratch parsing/validation by hand (Tasks 5,6,10) ✓; legal: no data shipped, synthetic examples, gitignore (Tasks 0.1, 0.2) ✓; manual-download/no-scraper (README, no scraper task exists) ✓; time-box ship-spine milestone (Task 12 note) ✓.

**Type consistency:** `call_model(prompt, client=None)` used uniformly (llm, planner, semantic, generator, judge); `embed_text(text, client=None)` (embed, rag); `QueryPlan{path, filters, semantic_intent, schema_blocked_reason}`, `Predicate{field, op, value}`, `Record{...}` consistent across executor/planner/pipeline; `answer_question(...) -> {path, doc_ids, answer, plan}` consumed identically in harness and compare; `doc_set_metrics/recall_at_k/mrr/schema_coverage/aggregate` signatures match call sites.

**Placeholder scan:** no TBD/TODO; every code step has complete code; prompts are concrete runnable templates (the documented iteration surface, not placeholders).
