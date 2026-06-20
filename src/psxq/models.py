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
