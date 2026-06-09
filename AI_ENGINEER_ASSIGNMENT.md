# AI Engineer — Take-Home

**Stack:** Python. Any vector store, embedding model, and LLM you like.
**Deadline:** One day from when you receive this. Pace it however you like.
**AI tools (Copilot / Cursor / ChatGPT):** allowed.

---

## Problem

Build a small **RAG** app — "Ask the Brand."

**Core**

- Generate (or use) a corpus of ~1,000–2,000 short documents about a fictional brand "Acme Co." — a mix of social posts and news snippets, with a few recurring themes, varied sentiment, and a couple of deliberately contradictory items. Each doc has at least: `id, source, posted_at, author, body, url, sentiment`.
- Ingest, chunk, embed, and index the corpus.
- Expose a way to ask questions (a `POST /ask` endpoint or a tiny Streamlit / Gradio page).
- Answers must be **grounded** in the corpus, include **citations** (doc id or url), and **refuse** when the corpus doesn't support an answer.

**A bit more**

- **Incremental ingestion** — a way to add new documents later without re-embedding the whole corpus. Show that a second ingest run only processes the new docs.
- **Metadata filters** in retrieval — the user (or the API) can constrain by `source`, `date range`, and `sentiment`. Show that the filter actually changes what gets retrieved.
- **Hybrid retrieval** — combine dense (vector) search with a keyword signal (BM25 / simple keyword match) and merge the results. Be ready to explain the merge.
- **Beyond plain Q&A** — support at least one of:
  - **summarization** over a filter ("summarize negative news from the last 30 days"), or
  - **comparison** ("how did sentiment about the new launch change before vs after April 10"),
  - i.e. the system should do more than retrieve-and-paraphrase a single chunk.
- A small **eval set** (10–15 hand-written Q&A pairs) and a script that runs them and prints retrieval hit-rate and whether the answer cited the right doc(s).

## Expected output

A public GitHub repo with a README that tells us how to run it and includes the corpus (or the script that generates it).

## Questions

If anything is unclear, ask. We'd rather answer a question than have you guess.

## Follow-up

We'll do a 30-min call. Walk us through the pipeline, defend your choices (chunking, model, top-k, prompt), and make a small change live.
