#!/usr/bin/env python
"""Ingest a JSONL corpus into Weaviate (incremental by default).

Examples:
    python scripts/run_ingest.py --path data/corpus.jsonl     # first run: all new
    python scripts/run_ingest.py --path data/corpus.jsonl     # again: all skipped
    python scripts/run_ingest.py --path data/new_docs.jsonl   # only the new docs processed
    python scripts/run_ingest.py --path data/corpus.jsonl --recreate  # wipe + reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ask_the_brand.ingest import ingest_file


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documents into Weaviate.")
    ap.add_argument("--path", required=True, help="Path to a JSONL file of documents.")
    ap.add_argument("--recreate", action="store_true",
                    help="Drop the collection + manifest and start fresh.")
    args = ap.parse_args()

    print(f"Ingesting {args.path} (recreate={args.recreate}) ...")
    result = ingest_file(args.path, recreate=args.recreate)
    print(
        f"  new={result['new']}  changed={result['changed']}  skipped={result['skipped']}\n"
        f"  chunks_embedded={result['chunks_embedded']}  "
        f"total_docs_in_manifest={result['total_docs_in_manifest']}"
    )
    if result["new"] == 0 and result["changed"] == 0:
        print("  (Nothing new to embed — incremental ingestion working as intended.)")


if __name__ == "__main__":
    main()
