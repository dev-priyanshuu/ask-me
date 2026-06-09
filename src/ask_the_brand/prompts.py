"""Prompt templates for the answer / summarize / compare chains."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

REFUSAL_MARKER = "INSUFFICIENT_CONTEXT"

_GROUNDING_RULES = (
    "You are the knowledge assistant for the brand \"Acme Co.\". Answer using ONLY the "
    "information in the CONTEXT below. Each context item is tagged with its source id like "
    "[doc_id].\n"
    "Rules:\n"
    "- Cite the id of every source you rely on, inline, in square brackets, e.g. [acme-0123].\n"
    "- Do NOT use any outside knowledge or make up facts or ids.\n"
    "- If the context does not contain enough information to answer, reply with exactly "
    f"'{REFUSAL_MARKER}: ' followed by one short sentence saying the corpus doesn't cover it.\n"
    "- If sources disagree, say so explicitly and cite each conflicting source."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _GROUNDING_RULES),
    ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nAnswer:"),
])

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _GROUNDING_RULES + "\nProduce a concise, well-structured summary grounded in the "
               "context. Group related points and note any disagreements."),
    ("human", "TASK:\n{instruction}\n\nCONTEXT (documents matching the requested filter):\n"
              "{context}\n\nSummary:"),
])

COMPARE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _GROUNDING_RULES + "\nYou are comparing how the discussion changed between two "
               "time periods. Describe the shift in sentiment and themes from BEFORE to AFTER, "
               "with specific citations, and give a one-line takeaway."),
    ("human", "TOPIC: {topic}\nSPLIT DATE: {split_date}\n\n"
              "CONTEXT — BEFORE {split_date} ({before_count} docs):\n{before_context}\n\n"
              "CONTEXT — ON/AFTER {split_date} ({after_count} docs):\n{after_context}\n\n"
              "Comparison:"),
])
