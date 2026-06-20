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
