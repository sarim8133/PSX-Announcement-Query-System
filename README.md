# PSX Announcement Query System

NL questions over PSX corporate-announcement extractions: question -> structured filter -> grounded answer, measured. Optional vector-RAG arm for an honest comparison. See `docs/superpowers/specs/`.

**Data is not shipped** (PSX prohibits redistribution). The repo ships synthetic examples in `data/synthetic/`; real PDFs/JSON stay local and gitignored. Scraping PSX is prohibited — source PDFs are downloaded manually.

## Run
`pip install -e ".[dev]"` then `pytest`. To run eval on real data: set `GEMINI_API_KEY`, point the harness at `data/real/`.
