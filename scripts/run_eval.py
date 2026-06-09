#!/usr/bin/env python
"""Run the eval set and print retrieval hit-rate, citation accuracy, and refusal accuracy.

Requires Weaviate running + the corpus ingested + OPENAI_API_KEY set.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ask_the_brand.rag import answer_question
from ask_the_brand.retrieval import hybrid_search
from ask_the_brand.schemas import Filters

EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "eval_set.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the RAG eval harness.")
    ap.add_argument("--path", default=str(EVAL_PATH))
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    spec = json.loads(Path(args.path).read_text())
    top_k = args.top_k or spec.get("top_k", 8)
    items = spec["items"]

    rows = []
    # Aggregates
    ret_hits = ret_total = cite_hits = cite_total = refuse_hits = refuse_total = 0

    for it in items:
        expected = set(it.get("expected_doc_ids", []))
        expects_refusal = bool(it.get("expects_refusal", False))
        filters = Filters(**it["filters"]) if it.get("filters") else None

        retrieved = hybrid_search(it["question"], filters=filters, top_k=top_k)
        retrieved_ids = [c.doc_id for c in retrieved]

        resp = answer_question(it["question"], filters=filters, top_k=top_k)
        cited_ids = {c.doc_id for c in resp.citations}

        # Retrieval hit-rate (only for items that expect specific docs).
        ret_hit = None
        if expected:
            ret_total += 1
            ret_hit = bool(expected & set(retrieved_ids))
            ret_hits += int(ret_hit)

        # Citation accuracy (only for items that expect specific docs).
        cite_hit = None
        if expected:
            cite_total += 1
            cite_hit = bool(expected & cited_ids)
            cite_hits += int(cite_hit)

        # Refusal accuracy (only for refusal items).
        refuse_ok = None
        if expects_refusal:
            refuse_total += 1
            refuse_ok = (resp.refused is True)
            refuse_hits += int(refuse_ok)

        rows.append({
            "id": it["id"],
            "expected": ",".join(sorted(expected)) if expected else ("REFUSE" if expects_refusal else "-"),
            "retrieved_top": ",".join(retrieved_ids[:3]),
            "ret_hit": ret_hit,
            "cite_hit": cite_hit,
            "refused": resp.refused,
            "refuse_ok": refuse_ok,
        })

    # ---- print per-item table ----
    def fmt(v):
        if v is None:
            return " - "
        return " ✓ " if v else " ✗ "

    print("\n" + "=" * 100)
    print(f"{'id':<28}{'ret':<6}{'cite':<6}{'refused':<9}{'refuse_ok':<11}expected")
    print("-" * 100)
    for r in rows:
        print(f"{r['id']:<28}{fmt(r['ret_hit']):<6}{fmt(r['cite_hit']):<6}"
              f"{str(r['refused']):<9}{fmt(r['refuse_ok']):<11}{r['expected']}")
    print("=" * 100)

    def pct(a, b):
        return f"{(100.0 * a / b):.1f}% ({a}/{b})" if b else "n/a"

    print("\nAGGREGATE METRICS")
    print(f"  Retrieval hit-rate (>=1 expected doc in top-{top_k}): {pct(ret_hits, ret_total)}")
    print(f"  Citation accuracy  (>=1 expected doc cited):         {pct(cite_hits, cite_total)}")
    print(f"  Refusal accuracy   (refused when it should):         {pct(refuse_hits, refuse_total)}")
    print()


if __name__ == "__main__":
    main()
