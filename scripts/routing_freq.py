"""
Per-question routing FREQUENCY across N runs, both prompts, at temperature 1.0.
Converts single-run routing observations into rates so stable changes (real effects)
can be separated from noisy ones (temp-1.0 instability).

Cost: 30 questions * N samples * 2 prompts. At N=10 that's 600 short calls.
"""
from __future__ import annotations
import json, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from psxq.llm import get_client, call_model
from psxq.planner import strip_fences

GOLD_FILE = "data/real/gold_questions.json"
TODAY = "2026-06-21"
N = 10
WORKERS = 6
STABLE = 0.8  # modal path must appear in >=80% of runs to count as "stable"

TEMPLATE = '''You translate a question about PSX corporate announcements into a JSON query plan.
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

{block}

Output ONLY JSON, no markdown fences:
{{"path": "...", "filters": [{{"field": "...", "op": "...", "value": ...}}],
  "semantic_intent": null, "schema_blocked_reason": null}}

Question: {question}
[nonce:{nonce}]'''

NEW_BLOCK = '''Choose exactly one path:
- "structured": the key distinction maps cleanly to field predicates, even if the question
  sounds broad. e.g. "fund companies distributed dividends" -> signals contains "Dividend";
  "Ghani ChemWorld right issue activity" -> signals contains "Right Issue".
- "semantic": the key distinction requires reading document text beyond what field predicates
  capture, even if some schema fields are adjacent. e.g. "extraordinary general meetings" ->
  the "General Meeting" signal does not distinguish ordinary vs extraordinary; needs text.
  e.g. "routine administrative noise" -> no field captures whether an announcement is
  meaningful; needs judgment.
- "schema_blocked": the answer is NOT representable in the schema (right/bonus issue
  ratio or subscription price, multi-tranche record dates, meeting outcomes, Shariah
  compliance). Set schema_blocked_reason. Do NOT invent fields.'''

OLD_BLOCK = '''Choose exactly one path:
- "structured": answerable by filtering the fields above.
- "semantic": genuinely fuzzy intent not expressible as field predicates.
- "schema_blocked": the answer is NOT representable in the schema (right/bonus issue
  ratio or subscription price, multi-tranche record dates, meeting outcomes, Shariah
  compliance). Set schema_blocked_reason. Do NOT invent fields.'''

PATHS = ("structured", "semantic", "schema_blocked")
ABBR = {"structured": "str", "semantic": "sem", "schema_blocked": "blk", "PARSE_ERR": "ERR"}

def get_path(raw: str) -> str:
    try:
        data = json.loads(strip_fences(raw))
        p = data.get("path")
        if p in PATHS:
            return p
    except Exception:
        pass
    m = re.search(r'"path"\s*:\s*"(structured|semantic|schema_blocked)"', raw or "")
    return m.group(1) if m else "PARSE_ERR"

def dist_str(counter: Counter) -> str:
    return " ".join(f"{ABBR.get(p,p)}={counter.get(p,0)}" for p in PATHS if counter.get(p, 0))

gold = json.loads(Path(GOLD_FILE).read_text(encoding="utf-8"))
client = get_client()

# Build all tasks: (variant, qid, gold_path, question, sample_idx)
tasks = []
for variant, block in (("old", OLD_BLOCK), ("new", NEW_BLOCK)):
    for q in gold:
        for i in range(N):
            tasks.append((variant, q["id"], q["gold_path"], q["question"], i, block))

def run_one(t):
    variant, qid, gold_path, question, i, block = t
    prompt = TEMPLATE.format(today=TODAY, block=block, question=question,
                             nonce=f"{time.time_ns()}-{i}")
    raw = call_model(prompt, client=client)
    return (variant, qid, get_path(raw))

results = {"old": {q["id"]: [] for q in gold}, "new": {q["id"]: [] for q in gold}}
done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(run_one, t) for t in tasks]
    for f in as_completed(futs):
        variant, qid, path = f.result()
        results[variant][qid].append(path)
        done += 1
        if done % 60 == 0:
            print(f"  ... {done}/{len(tasks)} calls done", flush=True)

gold_path = {q["id"]: q["gold_path"] for q in gold}
qtext = {q["id"]: q["question"] for q in gold}

def match_rate(variant, qid):
    paths = results[variant][qid]
    return paths.count(gold_path[qid]) / len(paths) if paths else 0.0

def modal(variant, qid):
    c = Counter(results[variant][qid])
    p, n = c.most_common(1)[0]
    return p, n / len(results[variant][qid])

# Expected routing accuracy = mean per-question match rate
def expected_acc(variant):
    return sum(match_rate(variant, q["id"]) for q in gold) / len(gold)

print("\n" + "=" * 78)
print(f"ROUTING FREQUENCY  (N={N} samples/question/prompt, temperature=1.0)")
print("=" * 78)

print(f"\nExpected routing accuracy vs gold (mean per-question match rate):")
print(f"  old prompt: {expected_acc('old'):.3f}")
print(f"  new prompt: {expected_acc('new'):.3f}")
print(f"  delta:      {expected_acc('new')-expected_acc('old'):+.3f}")

print("\n--- Per-question: gold | old(dist, match%) | new(dist, match%) | verdict ---")
flips, regress, noisy = [], [], []
for q in gold:
    qid = q["id"]
    gp = gold_path[qid]
    oc, nc = Counter(results["old"][qid]), Counter(results["new"][qid])
    omr, nmr = match_rate("old", qid), match_rate("new", qid)
    _, ostab = modal("old", qid)
    _, nstab = modal("new", qid)
    is_noisy = ostab < STABLE or nstab < STABLE
    verdict = ""
    if nmr - omr >= 0.3:
        verdict = "FIX (+%.0f%%)" % ((nmr - omr) * 100); flips.append(qid)
    elif omr - nmr >= 0.3:
        verdict = "REGRESS (-%.0f%%)" % ((omr - nmr) * 100); regress.append(qid)
    if is_noisy:
        verdict = (verdict + " [NOISY]").strip()
        noisy.append(qid)
    mark = "" if not verdict else "  <<"
    print(f"{qid:4s} {ABBR[gp]:3s} | old[{dist_str(oc):14s} {omr*100:3.0f}%] | "
          f"new[{dist_str(nc):14s} {nmr*100:3.0f}%] | {verdict}{mark}")

print("\n--- Summary ---")
print(f"Stable FIXES   (match rate +>=30pp): {flips}")
print(f"Stable REGRESS (match rate -<=30pp): {regress}")
nboth = sum(1 for q in gold if modal('old', q['id'])[1] < STABLE or modal('new', q['id'])[1] < STABLE)
print(f"NOISY questions (modal path <{int(STABLE*100)}% in some prompt): {len(set(noisy))} of {len(gold)} -> {sorted(set(noisy))}")
print(f"\nNet routing accuracy: old {expected_acc('old'):.3f} -> new {expected_acc('new'):.3f} "
      f"({expected_acc('new')-expected_acc('old'):+.3f})")
