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
