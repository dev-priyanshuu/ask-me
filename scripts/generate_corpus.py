#!/usr/bin/env python
"""Generate the Acme Co. corpus.

Examples:
    python scripts/generate_corpus.py                 # ~1500 docs via the LLM
    python scripts/generate_corpus.py --offline       # template bodies, no API calls
    python scripts/generate_corpus.py --limit 50      # quick smoke test
    python scripts/generate_corpus.py --force         # overwrite an existing corpus
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `import ask_the_brand` work whether or not the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ask_the_brand.config import settings
from ask_the_brand.corpus_generator import generate_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the Acme Co. corpus.")
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--new-out", default="data/new_docs.jsonl",
                    help="Where to write the incremental 'new docs' file.")
    ap.add_argument("--n", type=int, default=1500, help="Approx. number of docs in the main corpus.")
    ap.add_argument("--n-new", type=int, default=25, help="Number of incremental new docs.")
    ap.add_argument("--limit", type=int, default=None, help="Shortcut: small corpus for a smoke test.")
    ap.add_argument("--offline", action="store_true",
                    help="Use template bodies instead of the LLM (no OpenAI calls).")
    ap.add_argument("--force", action="store_true", help="Overwrite if the corpus already exists.")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists. Use --force to regenerate. (Skipping.)")
        return

    n = args.limit if args.limit is not None else args.n
    offline = args.offline or not settings.openai_api_key
    if offline and not args.offline:
        print("No OPENAI_API_KEY found -> falling back to --offline template generation.")

    print(f"Generating ~{n} docs (offline={offline}) -> {out}")
    result = generate_corpus(
        out_path=str(out),
        n_docs=n,
        offline=offline,
        new_docs_path=args.new_out,
        n_new=args.n_new,
    )
    print(f"Wrote {result.get('corpus')} docs to {out}")
    if "new_docs" in result:
        print(f"Wrote {result['new_docs']} incremental docs to {args.new_out}")


if __name__ == "__main__":
    main()
