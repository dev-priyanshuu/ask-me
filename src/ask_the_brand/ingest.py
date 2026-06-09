"""Chunk, embed, and (incrementally) index documents into Weaviate.

Incremental ingestion
----------------------
A manifest (``data/.ingest_manifest.json``) maps ``doc_id -> content_hash`` (sha256
of the body). On each run we only (re)embed docs that are **new** or whose body
**changed**; unchanged docs are skipped entirely (no embedding cost). Chunks use
deterministic UUIDs (``uuid5(doc_id:chunk_index)``) so re-ingesting a changed doc
upserts cleanly instead of creating duplicates.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .corpus_generator import load_jsonl
from .models import get_embeddings
from .vectorstore import ensure_collection, get_client

MANIFEST_PATH = Path("data/.ingest_manifest.json")
_NAMESPACE = uuid.UUID("a3c5f2b0-1d4e-4a6b-9c8d-000000000000")


def _content_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _chunk_uuid(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{chunk_index}"))


def _load_manifest() -> Dict[str, str]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: Dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _chunk_doc(doc: dict, splitter: RecursiveCharacterTextSplitter) -> List[Tuple[int, str]]:
    """Split a doc body into (chunk_index, text). Short docs -> a single chunk."""
    pieces = splitter.split_text(doc["body"]) or [doc["body"]]
    return list(enumerate(pieces))


def ingest_file(path: str, recreate: bool = False) -> Dict[str, int]:
    """Ingest a JSONL file incrementally. Returns counts."""
    docs = load_jsonl(path)
    manifest = {} if recreate else _load_manifest()
    splitter = _splitter()
    embeddings = get_embeddings()

    new, changed, skipped = 0, 0, 0
    pending: List[Tuple[dict, List[Tuple[int, str]]]] = []  # docs that need (re)embedding

    for doc in docs:
        h = _content_hash(doc["body"])
        prev = manifest.get(doc["id"])
        if prev is None:
            new += 1
        elif prev != h:
            changed += 1
        else:
            skipped += 1
            continue
        pending.append((doc, _chunk_doc(doc, splitter)))
        manifest[doc["id"]] = h

    chunks_embedded = 0
    with get_client() as client:
        coll = ensure_collection(client, recreate=recreate)

        # Embed all pending chunks in one batched call, then upsert.
        flat_texts: List[str] = []
        flat_meta: List[Tuple[dict, int, str]] = []  # (doc, chunk_index, text)
        for doc, chunks in pending:
            for ci, text in chunks:
                flat_texts.append(text)
                flat_meta.append((doc, ci, text))

        if flat_texts:
            vectors = embeddings.embed_documents(flat_texts)
            with coll.batch.dynamic() as batch:
                for (doc, ci, text), vec in zip(flat_meta, vectors):
                    batch.add_object(
                        uuid=_chunk_uuid(doc["id"], ci),
                        vector=vec,
                        properties={
                            "doc_id": doc["id"],
                            "source": doc["source"],
                            "author": doc["author"],
                            "sentiment": doc["sentiment"],
                            "url": doc["url"],
                            "posted_at": doc["posted_at"],
                            "body": text,
                            "chunk_index": ci,
                            "content_hash": _content_hash(doc["body"]),
                        },
                    )
            chunks_embedded = len(flat_texts)
            failed = coll.batch.failed_objects
            if failed:
                raise RuntimeError(f"{len(failed)} objects failed to import; first: {failed[0]}")

    _save_manifest(manifest)

    return {
        "new": new,
        "changed": changed,
        "skipped": skipped,
        "chunks_embedded": chunks_embedded,
        "total_docs_in_manifest": len(manifest),
    }
