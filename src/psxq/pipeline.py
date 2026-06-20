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
