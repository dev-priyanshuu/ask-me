"""Hybrid retrieval (dense vectors + BM25) with metadata filters.

Hybrid merge
------------
Weaviate runs the BM25 (keyword) and vector (dense) searches separately and fuses
them with ``relativeScoreFusion``: each search's scores are min-max normalized to
[0, 1], then combined as ``alpha * vector + (1 - alpha) * bm25``. So:

    alpha = 1.0  -> pure vector search
    alpha = 0.0  -> pure BM25 keyword search
    alpha = 0.5  -> equal blend (default)

We embed the query with the same OpenAI model used at ingest and pass both the raw
query text (for BM25) and the query vector (for dense) to a single hybrid call.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import List, Optional

from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

from .config import settings
from .models import get_embeddings
from .schemas import Filters, RetrievedChunk
from .vectorstore import get_collection, get_client


def _parse_dt(value: str, end_of_day: bool = False) -> datetime:
    """Parse an ISO date/datetime into a tz-aware datetime (UTC)."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        # Date-only string like "2026-04-10".
        d = datetime.strptime(value, "%Y-%m-%d")
        dt = datetime.combine(d.date(), time.max if end_of_day else time.min)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def build_filter(filters: Optional[Filters]) -> Optional[Filter]:
    """Translate our Filters model into a Weaviate Filter (ANDed clauses)."""
    if filters is None or filters.is_empty():
        return None

    clauses: List[Filter] = []
    if filters.sources:
        clauses.append(Filter.any_of(
            [Filter.by_property("source").equal(s) for s in filters.sources]
        ))
    if filters.sentiments:
        clauses.append(Filter.any_of(
            [Filter.by_property("sentiment").equal(s) for s in filters.sentiments]
        ))
    if filters.date_from:
        clauses.append(Filter.by_property("posted_at").greater_or_equal(_parse_dt(filters.date_from)))
    if filters.date_to:
        clauses.append(
            Filter.by_property("posted_at").less_or_equal(_parse_dt(filters.date_to, end_of_day=True))
        )

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return Filter.all_of(clauses)


def hybrid_search(
    query: str,
    filters: Optional[Filters] = None,
    top_k: Optional[int] = None,
    alpha: Optional[float] = None,
) -> List[RetrievedChunk]:
    """Run a single hybrid (BM25 + vector) query with optional metadata filters."""
    top_k = top_k or settings.default_top_k
    alpha = settings.default_alpha if alpha is None else alpha

    qvec = get_embeddings().embed_query(query)
    wfilter = build_filter(filters)

    with get_client() as client:
        coll = get_collection(client)
        res = coll.query.hybrid(
            query=query,
            vector=qvec,
            alpha=alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            filters=wfilter,
            limit=top_k,
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )

        out: List[RetrievedChunk] = []
        for obj in res.objects:
            p = obj.properties
            posted = p.get("posted_at")
            posted_str = posted.isoformat() if isinstance(posted, datetime) else str(posted)
            out.append(RetrievedChunk(
                doc_id=str(p.get("doc_id", "")),
                chunk_index=int(p.get("chunk_index", 0) or 0),
                source=str(p.get("source", "")),
                author=str(p.get("author", "")),
                posted_at=posted_str,
                sentiment=str(p.get("sentiment", "")),
                url=str(p.get("url", "")),
                body=str(p.get("body", "")),
                score=float(obj.metadata.score or 0.0),
            ))
    return out
