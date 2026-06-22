# PSX Announcement Query System

## In plain English (for non-technical readers)

This tool lets you ask everyday questions about Pakistan Stock Exchange company announcements —
*"which companies have a board meeting after June 20?"* — and get the answer, with its source, in
seconds.

**The idea.** Most such questions are really *filters* — a date, an amount, an event type — like
filtering a spreadsheet, not fuzzy "search." The popular AI approach ("RAG": let an AI read everything
and guess what's relevant) is actually worse for this: slower, fuzzier, and it can't explain itself or
admit when there is no answer. So this does the opposite — it turns your question into an exact filter,
and only calls on the AI for the genuinely fuzzy questions. It also knows when to say *"I can't answer
that"* instead of inventing something.

**Tested honestly, not just claimed.** I also built the popular AI-search version and ran a fair
head-to-head on the same questions. Mine scored **77%** vs **55%**. I had *predicted* the AI version
would win on the fuzzy questions — it didn't, and I report that plainly.

**The part that matters most.** Throughout, I kept catching and correcting my *own* mistakes — even
ones that made my results look better. One change first appeared to give a **+11%** improvement; I dug
in, found that half of it was a measurement fluke, and cut my own number in half. This happened seven
times, in both directions. The point is not a chart going up — anyone can show that — it is that the
numbers can be *trusted*, because I visibly break my own claims before publishing them.

*Technical write-up follows.*

---

Ask questions in plain English over structured extractions of Pakistan Stock Exchange (PSX) corporate
announcements — *"which companies have a board meeting after June 20?"*, *"which announcements are in a
closed period?"* — and get a grounded answer with its source documents. The question is translated into
structured filter predicates, executed in Python, and answered from the matching records.

## What this is — and the thesis

**Most questions over this corpus are filters, not semantic search.** Dates, event types, numeric
thresholds, boolean flags — they map cleanly onto schema fields, and a structured query answers them
exactly and explainably. Classic vector RAG (embed everything, retrieve top-k, generate) is the
reflexive choice for "ask questions over documents," but it is the wrong tool when the corpus is
already structured and the questions are predicates.

So the system is **structured-query-first**, with a three-path router:

- `structured` — the question maps to field predicates; answer by filtering.
- `semantic` — the distinction needs reading prose; answer by an LLM relevance pass.
- `schema_blocked` — the answer is not representable in the schema; say so honestly instead of hallucinating.

To keep "structured beats embeddings here" an *argument* rather than an assertion, the repo includes a
fairly-built pure vector-RAG arm and a head-to-head harness — the point is to **measure** whether
structured-first actually wins on the filter-shaped majority, not just claim it.

**What it demonstrates:** NL→structured-query translation with optional-fence-robust JSON parsing; a
router that knows when a question is unanswerable; an evaluation harness reporting 3-trial mean+range
(never single-run); and — the part worth reading — a measurement-audit discipline that caught and
corrected the project's own results five times (see [Results](#results-real-data-run-30-gold-questions)).

**Data is not shipped** (PSX prohibits redistribution). The repo ships synthetic examples in
`data/synthetic/`; real PDFs/JSON stay local and gitignored. Scraping PSX is prohibited — source PDFs
are downloaded manually. Design details: `docs/superpowers/specs/`.

## Quickstart

```
pip install -e ".[dev]"
pytest          # 42 tests; no API key needed (LLM calls are mocked)
```

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

## Results (real-data run, 30 gold questions)

**Bottom line:** a planner-prompt fix improves **routing +3.7pp** (sign-robust at N=10) and
**retrieval +4.5pp** (suggestive, n=3); **correctness +5.5pp**, but unproven at n=3 (ranges overlap).
The more valuable result is the **measurement audit** below — half of an initial *+11.1%* correctness
headline turned out to be a single-trial artifact, caught and corrected before publishing.

This section leads with that measurement audit rather than a score — because the most
useful result here was catching a favorable number that turned out to be half artifact,
and reporting the smaller true effect on purpose.

### Measurement audit: a +11.1% headline reduced to +5.5pp

A planner-prompt fix first appeared to lift answer correctness **+11.1%** (0.633 → 0.744).
Auditing the pipeline before trusting that number surfaced three problems:

1. **A mislabeled metric.** What an earlier comparison called "path accuracy" was actually
   `translation_exact` (exact doc-set match). Tracing the harness: `doc_set_metrics` reads
   `gold_doc_ids`, and `schema_coverage` reads the *system's* path — neither is
   path-classification accuracy, which came from a separate throwaway diagnostic.

2. **A relabel that changes nothing it appeared to.** Two gold questions (q15, q20) were
   corrected semantic→structured. But `gold_path` and `gold_filters` are read by **nothing**
   in `run_eval` — only `gold_doc_ids` and `gold_answer_facts` feed the metrics. So the
   relabel contributes **exactly 0.0** to the correctness gain by construction; it moves only
   the routing diagnostic, not the eval summary.

3. **A single-trial baseline (the original sin).** The 0.633 baseline was one trial; the 0.744
   result was a 3-trial mean. Re-running the *old* prompt at 3 trials gives a mean of **0.689**
   (range 0.633–0.733) — the single trial had landed at the bottom of the true range. **About
   half the apparent gain was variance regression off an unlucky single draw**, not the fix.

### Corrected results — separated by confidence level

Answer metrics are 3-trial full-pipeline means with [min–max]; routing is measured separately at
N=10 single-pass samples per question per prompt (temperature 1.0). Δ = new (contrastive) − old (pre-fix).

| Metric | old | new | Δ | Range relationship |
|--------|-----|-----|---|--------------------|
| routing accuracy (N=10) | 89.0% | 92.7% | +3.7pp | sign robust (q16/q21 near-deterministic); ≈±1.4pp sampling SE (a lower bound) |
| `translation_exact` (retrieval) | 71.1% [70.0–73.3] | 75.6% [73.3–76.7] | +4.5pp | meet at 73.3, **no interior overlap** |
| `answer_correct` | 68.9% [63.3–73.3] | 74.4% [70.0–76.7] | +5.5pp | **overlap** (shared 70.0–73.3) |
| `answer_grounded` | 94.4% [86.7–100] | 95.6% [93.3–96.7] | +1.2pp | overlap |

- **Retrieval — suggestive; ranges touch but don't separate.** `translation_exact` rises +4.5pp and
  the 3-trial ranges *touch* at 73.3% with no interior overlap (old maxes out exactly where new
  bottoms out) — cleaner than correctness (whose ranges overlap), but at n=3 there is no formal
  confidence interval, so it ranks *below* the routing result, whose positive sign is robust to the
  measured noise.
- **Routing — the fix *trades* two error types, netting positive.** The mechanism is the finding;
  the number is the summary. The contrastive prompt converts two stable structured↔semantic errors
  into correct semantic routes (q16: 0→100%, q21: 20→100%, both 10/10) — but in doing so destabilizes
  two previously-stable questions onto the semantic↔schema_blocked boundary (q19: 100→40%, q24:
  100→60%). That trade nets **+3.7pp** routing accuracy (89.0→92.7%, N=10). What is robust is the *sign*: the
  gain is driven by two near-deterministic shifts (q16 0→100%, q21 20→100%) that no plausible
  variance erases. The *magnitude* is softer — a binomial sampling SE across the per-question rates
  is ≈±1.4pp, but that is a *lower bound* (it treats the 26 questions seen at 0/10 or 10/10 as
  variance-free, which a 10-sample run does not prove), and a normal-form CI is the wrong shape for a
  sum of near-0/1 proportions at N=10. Direction trustworthy; magnitude ≈±1.4pp or somewhat more. An earlier *single-run*
  diagnostic had claimed q14/q29 regressed and the net was flat — the 10× refuted both: q14 is 10/10
  correct under both prompts, q29's modal route stays correct (drifts 2/10), and the true net is
  positive. Those single-draw "regressions" were noise; the real ones (q19, q24) were invisible until
  the rates.
- **Routing stability — non-determinism localizes to the fuzzy boundary.** At temperature 1.0,
  exactly 4 of 30 questions have unstable routes (modal path <80% across 10 runs): q18, q19, q22,
  q24 — *all on the semantic↔schema_blocked boundary.* Every structured question and every
  clear-cut case is 100% stable across all 10 runs. The instability is not diffuse; it concentrates
  on precisely the boundary the ceiling section describes.
- **Correctness — a directional effect, not noise and not proven.** The gain is suggestive
  (+5.5pp mean), mechanistically expected since routing and retrieval are upstream of correctness,
  but unproven at n=3 because the 3-trial ranges overlap (shared 70.0–73.3). This is a real
  directional signal with a causal story behind it that is not yet statistically clean — not a
  null result, and not a settled win.
- **Groundedness — high mean, but noisy and not interpreted as an effect.** The mean is high
  (95.6%), yet old₃'s range [86.7–100] spans 13 points on n=3 — the widest in the table. The
  +1.2pp Δ sits well inside that noise; the high mean does **not** imply stability, and no effect
  is claimed here.

*Reproducibility:* the diagnostic that produced the routing rates (`scripts/routing_freq.py`) is
public; the data it ran on stays local per PSX's terms — reproducible in method, not in raw data.

### Verdict

The planner fix **net-improves routing (+3.7pp at N=10; sign robust across the 10× batch, magnitude
≈±1.4pp sampling SE), with a suggestive retrieval gain (+4.5pp, n=3 ranges touch)**, at the cost of new instability
on the semantic↔schema_blocked boundary; the downstream **correctness gain is +5.5pp mean, consistent
with that mechanism but unproven at n=3** — not the +11.1% first reported. More trials (10–20) would
likely settle the correctness claim and would be the right move *for production*; for this project
the methodological lesson is already complete, so the marginal certainty would not change any
conclusion. Knowing when more data is worth collecting is itself part of the result.

The most valuable output of this evaluation was not a score but a habit: five times across both PSX
projects, the evaluation caught the evaluator's own claim and corrected it toward truth — *in both
directions*. Four were corrections of favorable or overconfident claims: in Project 1, that the gold
set itself was soft; in Project 2, a mislabeled metric, a single-trial baseline artifact, and an
unverified "deterministic routing" claim. The fifth went the other way. A single-run diagnostic
showed "errors migrated, net flat," and the cheap, careful-looking move was to report it and stop —
that was the instinct. Re-running it 10× anyway, against that cost instinct, refuted it: the supposed
regressions (q14/q29) were noise, and the true net was a *positive* +3.7pp. Correcting a pessimistic
read *upward* — not only inflated ones downward — is the tell that this is measurement, not performed
humility: anyone can revise numbers down to look careful; revising up *and* down toward whatever the
data says is the skill. The generalizable lesson: a noisy instrument demands rate-based measurement
even when you think you've already seen enough.

### Process lesson

`run_eval` already defaults to `--trials 3`; the violation was overriding it to `--trials 1` during
debugging and then comparing across that boundary. Rule going forward: **never establish a comparison
baseline at single-trial.**

## Known ceiling: the semantic / schema_blocked boundary

**Two independent signals converge on the same boundary.** The gold *labels* are contestable on the
semantic↔schema_blocked line — for several questions a human can defend either label (q18 and q22
are the clearest cases, argued below). And the *model* is unstable on that same line — across the
N=10 run, the only 4 questions that route differently run-to-run (q18, q19, q22, q24) all sit on it,
while all 26 other questions are 100% stable. Two different phenomena — human disagreement about the
right label, and the model's inability to commit to one — concentrate on the **same boundary**, with
q18/q22 sitting in *both* sets. A ceiling confirmed from two directions at once is a real property of
the problem — not a labeling artifact, and not a model artifact. The rest of this section is why.

Routing accuracy is capped below 100% **by construction**, because two of the three categories
overlap on a class of questions where no single label is correct even to a human annotator:

- `semantic` — the answer is in the announcement prose and a text-relevance pass can retrieve it.
- `schema_blocked` — the answer is not representable in the extraction schema.

These are not disjoint. Any fact that is *absent from the schema* but *present in the prose*
satisfies both definitions at once. Two gold questions sit exactly on this line:

- **q18** ("share buyback programme") — there is no `Buyback` signal type; the fact lives only in
  the announcement text. Routing it `semantic` (read the text) or `schema_blocked` (no field holds
  it) are both defensible.
- **q22** ("recurring *daily* fund distributions, not one-time") — the schema has no frequency
  field, so "daily vs one-time" cannot be filtered; yet the word "daily" is in the prose, so a text
  pass can find it.

The system routes both to `schema_blocked`; the gold labels them `semantic`. Both choices are
defensible, so any fixed gold label penalizes a system that picks the other. The only ways to remove
this ~2/30 loss are to (a) merge the two categories, or (b) impose an arbitrary tie-break rule the
gold set also adopts — neither reflects a real distinction. We treat it as a **structural floor, not
a model error**, and leave q18/q22 labeled `semantic`.

**Sharpening the prompt moved the boundary rather than resolving it.** q19 and q24 were *stable*
under the pre-fix prompt; they destabilized only once the contrastive prompt sharpened the
semantic/schema_blocked definitions. Tightening the language did not dissolve the ambiguity — it
changed *which* questions fall across the line. That is the signature of an irreducible distinction,
not a fixable prompt gap.

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

**Results** (spine: 3-trial mean [min–max] at temp 1.0; vector: single deterministic pass; k=4; 30 gold questions). Produced by `scripts/compare_trials.py`.

**The hypothesis was directionally right but quantitatively wrong — and that is the result worth leading with.** I predicted the vector arm would *tie or win* on the fuzzy semantic minority. It did not: it *narrowed* the gap (structured ~26pp → semantic ~14pp) but never overtook the spine on the honest same-scale metric. My own stated prediction was partly off, and the data says so plainly.

With that established, the measured payoff — doc-set F1, the one metric on the same scale for both arms:

| Arm | doc-set F1 | precision | recall | recall@4 | MRR |
|-----|-----------|-----------|--------|----------|-----|
| **spine** (3-trial) | **76.9% [74.1–79.0]** | 79.2% [76.7–81.5] | 79.2% [75.4–81.1] | — (set-based) | — |
| vector-RAG | 55.0% | 50.0% | 72.5% | 72.5% | 0.788 |

| Category | spine F1 (3-trial) | vector F1 | vector recall@4 |
|----------|--------------------|-----------|-----------------|
| structured | 79.8% [79.8–79.8] | 53.3% | 62.2% |
| semantic | 71.8% [64.2–77.8] | 58.0% | 90.6% |

Spine routes correctly **94.4% [90.0–96.7]** of the time and exactly matches the gold doc set on **74.4% [73.3–76.7]** of questions. It leads on F1 in both scored categories — by ~26pp on structured, ~14pp on semantic — the narrowed-but-never-overtaken gap above.

**recall@k flatters the vector arm exactly as warned — shown firing in the data.** On semantic, vector's recall@4 is **90.6%** (its single most impressive number) while its F1 is **58.0%** — a 33-point gap. recall@k rewards finding the gold doc *somewhere* in the top-4; F1 penalizes the wrong docs dragged along with it (a fixed k=4 return tanks precision when fewer docs are actually relevant). Headlining recall@k would have sold a "vector is competitive on semantic" story the same-scale F1 refutes — so F1 leads here, and recall@k is the cautionary secondary.

**The comparison independently reproduced the routing analysis's stability finding.** The spine's structured F1 has *zero* variance across trials [79.8–79.8] while its semantic F1 swings [64.2–77.8] — the same structured-stable / boundary-noisy split the routing analysis above found by an entirely unrelated measurement. Two different methods landing on the same split is converging evidence it is a real property of the task, not an artifact of either method.

**An architectural difference the F1 numbers do *not* capture.** schema_blocked questions have no gold docs, so they were *excluded* from F1/recall scoring for both arms. Outside those numbers there is a real structural difference: a pure top-k retriever cannot abstain — it always returns k documents — whereas the spine routes unanswerable questions to `schema_blocked` and returns nothing. I flag this as an architectural property, **not** a measured win, precisely because it sits outside the scored comparison.

**Bottom line:** my hypothesis was directionally right and quantitatively wrong — vector narrowed the gap on the fuzzy questions but never overtook the spine on the honest metric, and looked competitive only on the metric (recall@k) that flatters it. Structured-first was the right call for this corpus — measured where measurable, flagged as architectural where not.
