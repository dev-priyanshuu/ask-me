"""FastAPI app exposing the RAG pipeline.

Endpoints
---------
GET  /health      - Weaviate connectivity + indexed object count
POST /ingest      - incremental ingest of a JSONL file
POST /search      - raw hybrid retrieval (no LLM) — shows filters/alpha changing results
POST /ask         - grounded Q&A with citations + refusal
POST /summarize   - grounded summarization over a metadata filter
POST /compare     - before/after comparison around a split date
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import settings
from .ingest import ingest_file
from .rag import answer_question, compare, summarize
from .retrieval import hybrid_search
from .schemas import (
    AskRequest,
    AskResponse,
    CompareRequest,
    CompareResponse,
    Filters,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from .vectorstore import get_client, object_count

app = FastAPI(
    title="Ask the Brand — Acme Co. RAG",
    description="Grounded Q&A, summarization, and comparison over a synthetic Acme Co. corpus.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        with get_client() as client:
            ready = client.is_ready()
            count = object_count(client)
        return HealthResponse(status="ok", weaviate_ready=ready,
                              collection=settings.collection_name, object_count=count)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Weaviate unavailable: {exc}")


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    try:
        result = ingest_file(req.path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(path=req.path, **result)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    filters = req.filters or Filters()
    top_k = req.top_k or settings.default_top_k
    alpha = settings.default_alpha if req.alpha is None else req.alpha
    results = hybrid_search(req.query, filters=filters, top_k=top_k, alpha=alpha)
    return SearchResponse(query=req.query, alpha=alpha, top_k=top_k, filters=filters, results=results)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return answer_question(req.question, filters=req.filters, top_k=req.top_k, alpha=req.alpha)


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(req: SummarizeRequest) -> SummarizeResponse:
    return summarize(req.instruction, filters=req.filters, top_k=req.top_k)


@app.post("/compare", response_model=CompareResponse)
def compare_endpoint(req: CompareRequest) -> CompareResponse:
    return compare(req.topic, split_date=req.split_date, filters=req.filters, top_k=req.top_k)
