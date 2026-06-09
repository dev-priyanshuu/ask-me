"""Shared OpenAI client factories (LangChain wrappers)."""
from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env.")
    return OpenAIEmbeddings(model=settings.openai_embed_model, api_key=settings.openai_api_key)


@lru_cache
def get_chat_model(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env.")
    return ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )
