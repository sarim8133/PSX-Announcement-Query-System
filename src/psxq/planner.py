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
