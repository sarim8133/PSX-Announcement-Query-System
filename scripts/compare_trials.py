"""
Spine vs vector-RAG comparison, done to the project's rigor standard.

The spine routes non-deterministically (planner at temperature 1.0), so the spine arm is run
TRIALS times and reported as mean[min-max]. The vector arm is deterministic (fixed embeddings +
fixed top-k retrieval), so it is run once. Generation is skipped in both arms — the comparison
scores retrieved doc sets, not answer text — which halves cost vs the full `run_compare`.

Headline metric is doc-set F1 (same scale for both arms). recall@k / MRR are vector-only ranking
metrics, reported secondary. schema_blocked questions have no gold docs -> excluded from retrieval
scoring (their gold_doc_ids are empty).
"""
from __future__ import annotations
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from psxq.corpus import load_corpus
from psxq.llm import get_client
from psxq.planner import plan_query
from psxq.executor import execute
from psxq.semantic import semantic_filter
from psxq.vector.rag import build_index
from psxq.vector.embed import embed_text
from psxq.eval.metrics import doc_set_metrics, recall_at_k, mrr

RECORDS = "data/real/records.json"
GOLD = "data/real/gold_questions.json"
TODAY = "2026-06-21"
K = 4
TRIALS = 3
WORKERS = 6
CATS = ["structured", "semantic", "schema_blocked"]

records = load_corpus(RECORDS)
gold = json.loads(Path(GOLD).read_text(encoding="utf-8"))
client = get_client()

def spine_route(q):
    """Return (path, doc_ids) from routing+retrieval only — no answer generation."""
    plan = plan_query(q["question"], today=TODAY, client=client)
    if plan.path == "schema_blocked":
        sel = []
    elif plan.path == "semantic":
        cand = execute(plan, records) if plan.filters else records
        sel = semantic_filter(plan.semantic_intent or q["question"], cand, client=client)
    else:
        sel = execute(plan, records)
    return plan.path, [r.doc_id for r in sel]

index = build_index(records, client=client)

def vec_retrieve(q):
    return index.search(embed_text(q["question"], client=client), k=K)

# vector arm: deterministic, one pass
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    vec_ids = list(ex.map(vec_retrieve, gold))
print("vector pass done", flush=True)

# spine arm: non-deterministic, TRIALS passes
spine_trials = []
for t in range(TRIALS):
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        spine_trials.append(list(ex.map(spine_route, gold)))
    print(f"spine trial {t + 1}/{TRIALS} done", flush=True)

def mmr3(vals):
    vals = list(vals)
    return mean(vals), min(vals), max(vals)

# ---- spine per-trial aggregates, then mean[min-max] across trials ----
def spine_trial_metrics(routed):
    ex_, pa_, f_, p_, r_ = [], [], [], [], []
    for q, (path, ids) in zip(gold, routed):
        gids = q["gold_doc_ids"]
        m = doc_set_metrics(gids, ids)
        ex_.append(1.0 if m["exact"] else 0.0)
        pa_.append(1.0 if path == q["gold_path"] else 0.0)
        if gids:
            f_.append(m["f1"]); p_.append(m["precision"]); r_.append(m["recall"])
    return {"exact": mean(ex_), "path": mean(pa_), "f1": mean(f_), "prec": mean(p_), "rec": mean(r_)}

spt = [spine_trial_metrics(rt) for rt in spine_trials]
def sp(key): return mmr3(s[key] for s in spt)

# ---- vector aggregates (single pass) ----
vf, vp, vr, vrec, vmr = [], [], [], [], []
for q, ids in zip(gold, vec_ids):
    gids = q["gold_doc_ids"]
    if gids:
        m = doc_set_metrics(gids, ids[:K])
        vf.append(m["f1"]); vp.append(m["precision"]); vr.append(m["recall"])
        vrec.append(recall_at_k(ids, gids, K)); vmr.append(mrr(ids, gids))

# ---- by-category ----
def cat_spine_exact(cat):
    per_trial = []
    for rt in spine_trials:
        vals = [1.0 if doc_set_metrics(q["gold_doc_ids"], ids)["exact"] else 0.0
                for q, (path, ids) in zip(gold, rt) if q["gold_path"] == cat]
        if vals:
            per_trial.append(mean(vals))
    return mmr3(per_trial) if per_trial else None

def cat_spine_f1(cat):
    per_trial = []
    for rt in spine_trials:
        vals = [doc_set_metrics(q["gold_doc_ids"], ids)["f1"]
                for q, (path, ids) in zip(gold, rt)
                if q["gold_path"] == cat and q["gold_doc_ids"]]
        if vals:
            per_trial.append(mean(vals))
    return mmr3(per_trial) if per_trial else None

def cat_vec_f1(cat):
    vals = [doc_set_metrics(q["gold_doc_ids"], ids[:K])["f1"]
            for q, ids in zip(gold, vec_ids)
            if q["gold_path"] == cat and q["gold_doc_ids"]]
    return mean(vals) if vals else None

def cat_vec_recall(cat):
    vals = [recall_at_k(ids, q["gold_doc_ids"], K)
            for q, ids in zip(gold, vec_ids)
            if q["gold_path"] == cat and q["gold_doc_ids"]]
    return mean(vals) if vals else None

def fmt(t):
    return "n/a" if t is None else f"{t[0]*100:.1f}% [{t[1]*100:.1f}-{t[2]*100:.1f}]"

print("\n" + "=" * 70)
print(f"SPINE vs VECTOR  (spine: {TRIALS} trials mean[min-max]; vector: 1 pass, deterministic; k={K})")
print("=" * 70)
print("\n-- HEADLINE: doc-set F1 (same scale, both arms, gold-doc questions only) --")
print(f"  spine  F1: {fmt(sp('f1'))}")
print(f"  vector F1: {mean(vf)*100:.1f}%")
print("\n-- spine, full --")
print(f"  translation_exact (exact set): {fmt(sp('exact'))}")
print(f"  path_selection_accuracy:       {fmt(sp('path'))}")
print(f"  doc_set precision: {fmt(sp('prec'))}   recall: {fmt(sp('rec'))}")
print("\n-- vector, full (deterministic) --")
print(f"  doc_set precision: {mean(vp)*100:.1f}%  recall: {mean(vr)*100:.1f}%  F1: {mean(vf)*100:.1f}%")
print(f"  recall@{K}: {mean(vrec)*100:.1f}%   MRR: {mean(vmr):.3f}   (ranking metrics, vector-only)")
print("\n-- by category: spine F1 (3-trial) vs vector F1 / recall@k --")
for c in CATS:
    print(f"  {c:14s}  spine_F1 {fmt(cat_spine_f1(c)):22s}  vec_F1 "
          f"{('n/a' if cat_vec_f1(c) is None else f'{cat_vec_f1(c)*100:.1f}%'):8s}  "
          f"vec_recall@{K} {('n/a' if cat_vec_recall(c) is None else f'{cat_vec_recall(c)*100:.1f}%')}")
print(f"\n(schema_blocked has no gold docs -> excluded from F1/recall; spine still scored on exact-empty match)")
