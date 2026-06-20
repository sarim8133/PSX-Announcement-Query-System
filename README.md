# PSX Announcement Query System

NL questions over PSX corporate-announcement extractions: question -> structured filter -> grounded answer, measured. Optional vector-RAG arm for an honest comparison. See `docs/superpowers/specs/`.

**Data is not shipped** (PSX prohibits redistribution). The repo ships synthetic examples in `data/synthetic/`; real PDFs/JSON stay local and gitignored. Scraping PSX is prohibited — source PDFs are downloaded manually.

## Run
`pip install -e ".[dev]"` then `pytest`. To run eval on real data: set `GEMINI_API_KEY`, point the harness at `data/real/`.

## Running the evaluation

**Prerequisites**

1. Set your Gemini API key:
   ```
   export GEMINI_API_KEY=<your-key>
   ```
2. Place real data files at the following paths (gitignored; never committed):
   - `data/real/records.json` — corpus extracted from PSX announcements
   - `data/real/gold_questions.json` — gold question/answer set

**Command**

```bash
python -m psxq.run_eval \
  --records data/real/records.json \
  --gold data/real/gold_questions.json \
  --today 2026-06-20 \
  --trials 3
```

**Output**

The command prints a JSON summary containing:
- `summary.translation_exact` — mean/min/max fraction of questions where the structured filter retrieved exactly the gold doc set
- `summary.answer_correct` — mean/min/max fraction of answers judged correct against the gold facts
- `summary.answer_grounded` — mean/min/max fraction of answers judged grounded in the retrieved context
- `schema_coverage` — breakdown of which filter fields (ticker, date range, event type, etc.) were exercised across the gold set
- `trials` — number of independent runs averaged

Results will be recorded here after the real-data run.

## Spine vs vector-RAG comparison

**Purpose**

An honest head-to-head: the structured spine vs a fairly-built pure vector-RAG arm on the same questions. Both arms receive the same corpus and the same gold set; neither is tuned to win.

**Command**

```bash
python -m psxq.run_compare \
  --records data/real/records.json \
  --gold data/real/gold_questions.json \
  --today 2026-06-20 \
  --k 4
```

**What it reports**

The command prints a single JSON object containing:
- `spine.translation_exact` — fraction of questions where the structured filter retrieved exactly the gold doc set
- `spine.path_selection_accuracy` — fraction of questions where the correct answer field was selected
- `spine.doc_set` — precision/recall/f1 of the spine's predicted doc set against gold (directly comparable with vector)
- `vector.recall_at_k` — fraction of gold docs found in the top-k retrieved chunks (schema_blocked questions excluded from retrieval scoring)
- `vector.mrr` — mean reciprocal rank of the first relevant chunk (schema_blocked questions excluded)
- `vector.doc_set` — precision/recall/f1 of the vector arm's top-k doc set against gold (directly comparable with spine)
- `by_category` — per-question-category breakdown of both arms, including `vector_mrr` per category
- `note` — explanation of the metric distinction

**Metric honesty**

The spine is set-based: it returns an exact predicted set of documents, so its natural metric is exact-set match and doc-set precision. The vector arm is ranked top-k: its natural metrics are recall@k and MRR, which give credit for a gold document appearing anywhere in the top-k. To make the comparison directly honest, the harness also computes doc-set precision/recall/f1 for both arms over the same scored questions — these are on the same scale and can be read side-by-side. Note that recall@k tends to flatter the vector arm relative to doc-set recall, because it awards full credit even when the gold doc appears last in the top-k list.

**Hypothesis**

The structured spine should win on filter-shaped questions (ticker lookups, date-range filters, numeric comparisons) where the query maps cleanly onto schema fields. The vector arm should tie or win on the genuinely fuzzy minority — open-ended questions where the answer is buried in prose rather than a structured field.

**Honesty note**

At ~30 docs the "vector index" is numpy cosine similarity — there is no Faiss, no chunking pipeline, no production infrastructure. The contribution is the measured comparison and the judgment it enables, not the retrieval infrastructure.

Results will be recorded here after the real-data run.
