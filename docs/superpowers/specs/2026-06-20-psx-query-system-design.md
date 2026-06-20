# Project 2 — PSX Announcement Query System: Design

> **Status:** Design, awaiting review.
> **Date:** 2026-06-20
> **Relationship to Project 1:** Builds directly on P1's structured JSON extractions. Its output *is* this project's corpus. See `PROJECT_CONTEXT.md`.

---

## 1. What this project is (one paragraph)

A natural-language query system over PSX corporate announcements. A user asks a question in English; the system translates it into a **structured query** over Project 1's JSON extractions, executes it, and generates a grounded answer. Semantic search is scoped to the genuinely fuzzy minority of questions — it is not the spine. The point of the project is **judgment plus evaluation**: demonstrating the rare skill of *matching the tool to the problem* (recognizing these questions are mostly structured filters, where embeddings would hurt), and measuring that claim rigorously. The evaluation through-line from Project 1 — separate metrics, full attribution, mean + range over trials — carries straight over.

**Why not "classic RAG":** The real questions over this corpus (see §4) are overwhelmingly structured filters — date ranges, numeric thresholds, company predicates, booleans. Vector similarity is the wrong primitive for those. The defensible, harder-to-fake move is to build what the data wants and *prove* the obvious approach is wrong here, rather than perform RAG on a problem that doesn't need it.

**Stack:** Python, `google-genai` SDK, Gemini 2.5 Flash, Pydantic for hand-rolled validation, raw API calls. Same from-scratch constraint as P1 — no LangChain/LlamaIndex. We write the query parsing, validation, and evaluation by hand because that is the skill being learned.

---

## 2. Scope decision: Path A vs Path A+

The project has a spine (Path A) and an optional additive arm (Path A+). **The spine is identical either way**, so the decision does not block starting.

- **Path A (lean):** structured-query spine + a small LLM-judged semantic path for fuzzy questions. No embeddings, no vector index. Honest and complete at 15–30 docs.
- **Path A+:** everything in A, plus a genuinely-built vector-RAG arm run head-to-head against the spine on the same questions — built fairly, as if we wanted it to win. Ticks the "RAG / embeddings / vector DB" tooling box *and* turns the judgment into a measured result.

**Gating action (done by author before/around the vector arm, not before the spine):** check 5 real target job postings. If a meaningful share explicitly name RAG / embeddings / vector DBs → do A+. If not → ship lean A. This follows `PROJECT_CONTEXT.md` §7.3 ("pick based on real job postings").

---

## 3. Corpus

- **Source:** Project 1's JSON extractions. **Two distinct PSX legal constraints (both verified), each mapped to one design choice:**
  1. **Automated scraping is prohibited** → PDFs are **downloaded manually** from the portal by the author. No scraper, ever. Corpus size is bounded by hand-download (~30 docs).
  2. **Commercial redistribution is prohibited** (PSX asserts a proprietary compilation of its data) → we **publish code and methodology, not data.** The source PDFs **and the extracted JSON** both stay gitignored. The repo *describes* the data and ships **synthetic examples** for anything that must be runnable.
- **Size:** **~30 documents (locked).** Start with the existing 15-doc gold set and extend to ~30 by running P1's *existing* extractor over manually-downloaded PDFs already on hand. 30 gives filters enough rows to return varied, non-trivial result sets while staying hand-verifiable.
- **Record shape:** one record per announcement = P1's 12-field schema (company, listing/document dates, signals, is_actionable, meeting/closure/financial details) **plus** a synthesized text blob (subject + signals + one-line summary) for the semantic path.
- **Integrity:** gold-set integrity is sacred (PROJECT_CONTEXT §7). The query gold set (§6) is verified against these records the same way P1's labels were. Because the query gold set embeds real data (relevant-doc labels, gold answer facts), it is **also gitignored**; the repo ships a small **synthetic** gold set so the harness is runnable.

---

## 4. The questions (drives everything)

Representative target questions, tagged by the path they *should* take:

| # | Question | Intended path |
|---|---|---|
| 1 | Board meetings in the next 2 weeks to consider a dividend? | structured |
| 2 | Cash dividends this month with payout > PKR 5/share + record dates? | structured |
| 3 | Book closure start/end for [Company X]? | structured |
| 4 | Which companies entered a closed period this week? | structured |
| 5 | All announcements from [Company X] last quarter? | structured |
| 6 | Recent right/bonus issues — ratio and subscription price? | structured (which docs) + **schema-blocked** (ratio/price) |
| 7 | Dividend record dates still in the future (still actionable)? | structured (boolean) |
| 8 | Any general meeting that rejected/failed a resolution recently? | **schema-blocked** (outcome not captured) |
| 9 | [Company X]'s dividend history over the period? | structured (aggregation) |
| 10 | Upcoming entitlements falling in the same week? | structured (aggregation) |

Three path categories: **structured**, **semantic** (genuinely fuzzy intent), and **schema-blocked** — questions whose answer is *not representable* in P1's v1 schema (the documented gaps: corporate-action details like right/bonus ratios and subscription prices, multi-tranche record dates, meeting outcomes, Shariah compliance). The correct behavior on a schema-blocked question is to recognize it and answer *"not captured"* — never to extrapolate (P1 faithfulness rule).

The distribution itself is a finding: ~majority structured, a real minority semantic, and a measurable slice schema-blocked. **How often each path is needed — and especially "what fraction of real questions my schema simply cannot answer" — is a measured result, not an assumption.** This is the honest ceiling story carried over from P1 (§6 "known ceiling").

---

## 5. Architecture (the spine)

Five small, independently-testable units:

1. **`corpus`** — load P1 JSON into a list of validated records (Pydantic). Flatten nested fields for filtering. In-memory; no DB needed at this scale (a "vector DB" here would be numpy — we say so honestly).
2. **`query_planner` — this is the hard part of the project, and where the failures will concentrate.** NL question → a **validated `QueryPlan`** via a raw Gemini call. Translating fuzzy natural language into *precise* structured predicates (right field, right operator, right date window, right signal-set membership) is the error-prone step; it also classifies the question's path (structured / semantic / schema-blocked, §4). The plan is a structured filter spec (predicates: equality, ranges, date windows, set-membership on signals, booleans) plus an optional `semantic_intent` string and a `schema_blocked` flag with the reason. We strip fences, parse JSON, and validate against a Pydantic `QueryPlan` model **by hand** (P1 ethos, decision §2/§3). No `response_schema` auto-enforcement. The bulk of evaluation effort and iteration belongs here.
3. **`executor` — mechanical by design.** Apply the validated `QueryPlan` to the records: structured predicates run in Python (precise, deterministic — no LLM). If `semantic_intent` is present:
   - **Lean A:** an LLM relevance pass over the (already filtered) candidate records — honest at 30 docs, no index.
   - **A+:** embedding similarity ranking (see §7).

   The executor holds little risk — if it misbehaves it's a plain bug, not a judgment failure. Keeping it dumb is deliberate: it isolates the interesting failures in the planner.
4. **`generator`** — question + result records → grounded NL answer. Faithfulness rule inherited verbatim from P1: every claim must trace to a record; if the data isn't present, say so; never extrapolate.

Data flow: `question → query_planner → QueryPlan → executor → record set → generator → grounded answer`.

Error handling: malformed plan JSON → re-parse with optional-fence stripping (P1's recurring bug); invalid plan → surfaced as an eval failure, never silently coerced. Out-of-schema filter fields are recorded as planner errors, not crashes (P1 decision §5: measure, don't enforce).

---

## 6. Evaluation harness (the centerpiece)

A hand-built question gold set of **~30 questions**, each labeled with: intended path, gold structured filter (or gold relevant-doc set), and gold answer facts. Metrics are kept **separate** so failures attribute cleanly (P1's field-level-attribution discipline). They are split so the **spine is a complete, measured system on two metrics**, and the **vector arm adds the other two** — you reach a working, shippable system inside the time-box before the arm exists.

**Spine metrics (ship here — a complete project):**
1. **Query-translation accuracy** — does the generated filter return the correct doc set? Compared as a **set** (order-independent), reusing P1's set-compare idea. (This is where most failures land — see §5, the planner.)
2. **Answer correctness + faithfulness** — answer matches gold facts AND every claim is grounded in a retrieved record. LLM-judge anchored to hand checks.
   - Plus the **schema-coverage finding** (§4): the fraction of questions that are structured / semantic / schema-blocked — measured, not assumed. The "what my schema can't answer" ceiling.

**Arm metrics (added with the vector arm, Path A+):**
3. **Retrieval quality** — **Recall@k / Hit-rate@k (+ MRR)** against labeled relevant docs. Only meaningful once there's a ranked retriever (the vector arm, and the semantic path it competes with).
4. **Path-selection accuracy** — did the planner route to the right path? Becomes consequential once routing between structured and a real retrieval path is a live choice worth measuring.

**Protocol (inherited from P1):** 3 trials, report **mean + range** everywhere, never single-run numbers. Temp 1.0 with a `time.time_ns()` nonce. Change one thing at a time.

---

## 7. The vector-RAG arm (Path A+ only)

Built **fairly** — as if we wanted it to win, so the comparison is honest:

- **`vector_arm`** — embed the synthesized text representation (Gemini embedding model) into a real index. At ~30 docs the index is numpy cosine; optionally FAISS/Chroma to tick the literal tooling box, stated honestly as overkill-for-scale. Pure RAG pipeline: `question → embed → top-k cosine → generate`.
- **`compare`** — run the **same question gold set** through both arms; score retrieval quality and answer correctness **separately**, broken down by question category (structured-filter vs fuzzy-semantic).

**Hypothesis to test (not assume):** structured spine wins on filter questions; the vector arm ties or wins on the fuzzy minority. Reporting that the obvious approach loses on filters — *with numbers* — is the headline result. This demonstrates we can build RAG *and* have the judgment to know when not to.

**Honesty guardrails (mirror P1's "zero variance" framing):** don't oversell a 30-vector cosine search as a "vector database"; the skill on display is the comparison and the judgment, not the infrastructure.

---

## 8. Sequencing & time-box (the plan, in order)

1. **(Author) Confirm the checkbox** — 5 real postings; do they name RAG/vectors? Yes → A+, no → lean A. Does **not** block step 2.
2. **Build the structured-query spine, end to end, measured.** A complete, shippable project on its own (§5 + §6).
3. **Add the vector-RAG arm**, built fairly (§7).
4. **Run the comparison harness** — same questions, both arms, retrieval-quality and answer-correctness measured separately (§6).
5. **Time-box.** If Project 3's timeline is threatened, **ship the spine** and label the vector arm "future work." The spine stands alone; nothing about it depends on the arm.

---

## 9. Non-goals (YAGNI)

- No scraper, no corpus scaling, no real-time feed — **automated scraping is prohibited** (PDFs downloaded manually).
- **No shipping data** — commercial redistribution is prohibited, so PDFs *and* extracted JSON are gitignored; the repo ships **synthetic examples** only and describes the real data in prose. Both are hard constraints, not preferences (see §3).
- No framework (LangChain/LlamaIndex) — from-scratch is the point.
- No production deployment / UI (that's a Project 3 candidate).
- Not solving P1's known schema gaps (multi-tranche, corporate_action, Shariah field, meeting outcomes) — questions that hit those are *recognized and answered "not captured,"* not extracted. Honest limitation, not a bug.

---

## 10. Resolved decisions

1. **Corpus size:** **~30 documents**, extended from the 15-doc gold set via the existing P1 extractor over manually-downloaded PDFs. (Scraping is not legal — see §3/§9.)
2. **Plan representation:** **structured filter-spec executed in Python.** The planner emits a Pydantic-validated `QueryPlan`; predicates run in Python over the in-memory records. No NL→SQL — it fits the nested schema and the from-scratch ethos better.
3. **Gold question count:** start at **~30 questions** spanning the path categories in §4; expand toward 50 only if category coverage proves thin.
