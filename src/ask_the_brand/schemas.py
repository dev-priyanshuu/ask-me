"""Pydantic models shared by the API and the core library."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
class Filters(BaseModel):
    """Metadata constraints applied during retrieval.

    All fields are optional; only the provided ones are ANDed together.
    """

    sources: Optional[List[str]] = Field(
        default=None, description="Restrict to these sources, e.g. ['news', 'twitter']."
    )
    sentiments: Optional[List[str]] = Field(
        default=None, description="Restrict to these sentiments, e.g. ['negative']."
    )
    date_from: Optional[str] = Field(
        default=None, description="Inclusive lower bound on posted_at (ISO date, e.g. 2026-04-10)."
    )
    date_to: Optional[str] = Field(
        default=None, description="Inclusive upper bound on posted_at (ISO date)."
    )

    def is_empty(self) -> bool:
        return not any([self.sources, self.sentiments, self.date_from, self.date_to])


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    doc_id: str
    chunk_index: int
    source: str
    author: str
    posted_at: str
    sentiment: str
    url: str
    body: str
    score: float


class Citation(BaseModel):
    doc_id: str
    url: str
    source: str
    posted_at: str
    sentiment: str
    score: float


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    query: str
    filters: Optional[Filters] = None
    top_k: Optional[int] = None
    alpha: Optional[float] = Field(
        default=None, description="Hybrid weight: 1.0 = pure vector, 0.0 = pure BM25."
    )


class AskRequest(BaseModel):
    question: str
    filters: Optional[Filters] = None
    top_k: Optional[int] = None
    alpha: Optional[float] = None


class SummarizeRequest(BaseModel):
    instruction: str = Field(
        default="Summarize the most important points.",
        description="What to summarize, e.g. 'Summarize the negative news from the last 30 days.'",
    )
    filters: Optional[Filters] = None
    top_k: Optional[int] = None


class CompareRequest(BaseModel):
    topic: str = Field(description="What to compare, e.g. 'the Acme Rocket Skates launch'.")
    split_date: Optional[str] = Field(
        default=None, description="ISO date separating 'before' from 'after'. Defaults to launch date."
    )
    filters: Optional[Filters] = None
    top_k: Optional[int] = None


class IngestRequest(BaseModel):
    path: str = Field(description="Path to a JSONL file of documents to ingest.")


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class SearchResponse(BaseModel):
    query: str
    alpha: float
    top_k: int
    filters: Filters
    results: List[RetrievedChunk]


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: List[Citation]
    filters: Filters
    alpha: float
    top_k: int


class SummarizeResponse(BaseModel):
    instruction: str
    summary: str
    refused: bool
    citations: List[Citation]
    filters: Filters
    n_docs: int


class CompareResponse(BaseModel):
    topic: str
    split_date: str
    comparison: str
    refused: bool
    before_count: int
    after_count: int
    citations: List[Citation]


class IngestResponse(BaseModel):
    path: str
    new: int
    changed: int
    skipped: int
    chunks_embedded: int
    total_docs_in_manifest: int


class HealthResponse(BaseModel):
    status: str
    weaviate_ready: bool
    collection: str
    object_count: int
