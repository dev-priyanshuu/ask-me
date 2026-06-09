"""Weaviate connection + the `AcmeDoc` collection schema.

We run Weaviate locally (docker-compose) with no vectorizer module: vectors are
produced by OpenAI embeddings via LangChain on the client side and written
explicitly. The `body` property is BM25-indexed so Weaviate can do the keyword
half of hybrid search.
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import weaviate
import weaviate.classes.config as wvc

from .config import settings


def connect() -> weaviate.WeaviateClient:
    """Open a connection to the local Weaviate instance."""
    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
    )


@contextlib.contextmanager
def get_client() -> Iterator[weaviate.WeaviateClient]:
    """Context manager that always closes the client (avoids gRPC warnings)."""
    client = connect()
    try:
        yield client
    finally:
        client.close()


def ensure_collection(client: weaviate.WeaviateClient, recreate: bool = False):
    """Create the AcmeDoc collection if it does not exist; return it."""
    name = settings.collection_name
    if recreate and client.collections.exists(name):
        client.collections.delete(name)

    if not client.collections.exists(name):
        client.collections.create(
            name=name,
            # We supply our own vectors (OpenAI embeddings), so use self-provided vectors.
            vector_config=wvc.Configure.Vectors.self_provided(),
            inverted_index_config=wvc.Configure.inverted_index(
                # BM25 tuning for the keyword half of hybrid search.
                bm25_b=0.75,
                bm25_k1=1.2,
            ),
            properties=[
                wvc.Property(name="doc_id", data_type=wvc.DataType.TEXT,
                             tokenization=wvc.Tokenization.FIELD),
                wvc.Property(name="source", data_type=wvc.DataType.TEXT,
                             tokenization=wvc.Tokenization.FIELD),
                wvc.Property(name="author", data_type=wvc.DataType.TEXT,
                             tokenization=wvc.Tokenization.FIELD),
                wvc.Property(name="sentiment", data_type=wvc.DataType.TEXT,
                             tokenization=wvc.Tokenization.FIELD),
                wvc.Property(name="url", data_type=wvc.DataType.TEXT,
                             tokenization=wvc.Tokenization.FIELD),
                wvc.Property(name="content_hash", data_type=wvc.DataType.TEXT,
                             tokenization=wvc.Tokenization.FIELD),
                # The text we search over (BM25 + embedded).
                wvc.Property(name="body", data_type=wvc.DataType.TEXT),
                wvc.Property(name="posted_at", data_type=wvc.DataType.DATE),
                wvc.Property(name="chunk_index", data_type=wvc.DataType.INT),
            ],
        )
    return client.collections.get(name)


def get_collection(client: weaviate.WeaviateClient):
    return client.collections.get(settings.collection_name)


def object_count(client: weaviate.WeaviateClient) -> int:
    try:
        coll = client.collections.get(settings.collection_name)
        return coll.aggregate.over_all(total_count=True).total_count
    except Exception:
        return 0
