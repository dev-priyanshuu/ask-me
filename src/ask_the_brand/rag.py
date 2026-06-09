"""RAG chains: grounded answer, filtered summarization, and before/after comparison.

Every chain retrieves from Weaviate first, formats the chunks with their [doc_id]
tags, and asks the LLM to answer ONLY from that context and cite the ids it used.
Refusal happens two ways: a pre-LLM score gate (nothing relevant retrieved) and the
LLM's own ``INSUFFICIENT_CONTEXT`` marker.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser

from .config import settings
from .models import get_chat_model
from .prompts import ANSWER_PROMPT, COMPARE_PROMPT, REFUSAL_MARKER, SUMMARIZE_PROMPT
from .retrieval import hybrid_search
from .schemas import (
    AskResponse,
    Citation,
    CompareResponse,
    Filters,
    RetrievedChunk,
    SummarizeResponse,
)

_CITE_RE = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def _format_context(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for c in chunks:
        lines.append(
            f"[{c.doc_id}] (source={c.source}, posted_at={c.posted_at[:10]}, "
            f"sentiment={c.sentiment})\n{c.body}"
        )
    return "\n\n".join(lines)


def _cited_ids(answer: str) -> List[str]:
    seen, out = set(), []
    for m in _CITE_RE.findall(answer):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _citations_from(answer: str, chunks: List[RetrievedChunk]) -> List[Citation]:
    """Map ids cited in the answer back to retrieved chunk metadata (best chunk per doc)."""
    best: Dict[str, RetrievedChunk] = {}
    for c in chunks:
        if c.doc_id not in best or c.score > best[c.doc_id].score:
            best[c.doc_id] = c
    cited = _cited_ids(answer)
    out: List[Citation] = []
    for doc_id in cited:
        c = best.get(doc_id)
        if c:  # ignore ids the model invented that weren't retrieved
            out.append(Citation(doc_id=c.doc_id, url=c.url, source=c.source,
                                 posted_at=c.posted_at, sentiment=c.sentiment, score=c.score))
    return out


def _is_refusal(answer: str) -> bool:
    return answer.strip().startswith(REFUSAL_MARKER)


# --------------------------------------------------------------------------- #
# Answer
# --------------------------------------------------------------------------- #
def answer_question(
    question: str,
    filters: Optional[Filters] = None,
    top_k: Optional[int] = None,
    alpha: Optional[float] = None,
) -> AskResponse:
    top_k = top_k or settings.default_top_k
    alpha = settings.default_alpha if alpha is None else alpha
    filters = filters or Filters()

    chunks = hybrid_search(question, filters=filters, top_k=top_k, alpha=alpha)

    # Pre-LLM refusal gate: nothing cleared the relevance threshold.
    if not chunks or max(c.score for c in chunks) < settings.refusal_score_threshold:
        return AskResponse(
            question=question,
            answer="I don't have enough information in the corpus to answer that.",
            refused=True, citations=[], filters=filters, alpha=alpha, top_k=top_k,
        )

    chain = ANSWER_PROMPT | get_chat_model(temperature=0.0) | StrOutputParser()
    raw = chain.invoke({"question": question, "context": _format_context(chunks)})

    refused = _is_refusal(raw)
    display = raw
    if refused:
        # Turn the marker into a friendly message.
        display = "I don't have enough information in the corpus to answer that."
    return AskResponse(
        question=question,
        answer=display,
        refused=refused,
        citations=[] if refused else _citations_from(raw, chunks),
        filters=filters, alpha=alpha, top_k=top_k,
    )


# --------------------------------------------------------------------------- #
# Summarize over a filter
# --------------------------------------------------------------------------- #
def summarize(
    instruction: str,
    filters: Optional[Filters] = None,
    top_k: Optional[int] = None,
) -> SummarizeResponse:
    top_k = top_k or settings.summary_top_k
    filters = filters or Filters()

    # Use the instruction itself as the retrieval query so the summary is focused.
    chunks = hybrid_search(instruction, filters=filters, top_k=top_k, alpha=settings.default_alpha)

    if not chunks:
        return SummarizeResponse(
            instruction=instruction,
            summary="No documents match that filter, so there is nothing to summarize.",
            refused=True, citations=[], filters=filters, n_docs=0,
        )

    chain = SUMMARIZE_PROMPT | get_chat_model(temperature=0.2) | StrOutputParser()
    raw = chain.invoke({"instruction": instruction, "context": _format_context(chunks)})

    refused = _is_refusal(raw)
    n_docs = len({c.doc_id for c in chunks})
    return SummarizeResponse(
        instruction=instruction,
        summary=("Nothing in the filtered set supports a summary." if refused else raw),
        refused=refused,
        citations=[] if refused else _citations_from(raw, chunks),
        filters=filters, n_docs=n_docs,
    )


# --------------------------------------------------------------------------- #
# Compare before / after a date
# --------------------------------------------------------------------------- #
def _split_by_date(chunks: List[RetrievedChunk], split: datetime) -> Tuple[List, List]:
    before, after = [], []
    for c in chunks:
        try:
            dt = datetime.fromisoformat(c.posted_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        (before if dt < split else after).append(c)
    return before, after


def compare(
    topic: str,
    split_date: Optional[str] = None,
    filters: Optional[Filters] = None,
    top_k: Optional[int] = None,
) -> CompareResponse:
    split_date = split_date or settings.launch_date
    top_k = top_k or settings.summary_top_k
    filters = filters or Filters()

    from .retrieval import _parse_dt  # local import to avoid cycle at module load
    split_dt = _parse_dt(split_date)

    # Retrieve a broad set about the topic, then split locally by posted_at.
    chunks = hybrid_search(topic, filters=filters, top_k=top_k, alpha=settings.default_alpha)
    before, after = _split_by_date(chunks, split_dt)

    if not before and not after:
        return CompareResponse(
            topic=topic, split_date=split_date,
            comparison="No documents about that topic were found, so there is nothing to compare.",
            refused=True, before_count=0, after_count=0, citations=[],
        )

    chain = COMPARE_PROMPT | get_chat_model(temperature=0.2) | StrOutputParser()
    raw = chain.invoke({
        "topic": topic,
        "split_date": split_date,
        "before_count": len(before),
        "after_count": len(after),
        "before_context": _format_context(before) or "(no documents before the split date)",
        "after_context": _format_context(after) or "(no documents on/after the split date)",
    })

    refused = _is_refusal(raw)
    return CompareResponse(
        topic=topic, split_date=split_date,
        comparison=raw,
        refused=refused,
        before_count=len(before), after_count=len(after),
        citations=[] if refused else _citations_from(raw, chunks),
    )
