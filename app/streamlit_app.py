"""Streamlit UI for 'Ask the Brand'. Talks to the FastAPI server over HTTP.

Run the API first:  uvicorn ask_the_brand.api:app --port 8000  (PYTHONPATH=src)
Then:               streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from ask_the_brand.config import settings
    DEFAULT_API = settings.api_base_url
except Exception:  # pragma: no cover
    DEFAULT_API = os.getenv("API_BASE_URL", "http://localhost:8000")

SOURCES = ["twitter", "reddit", "forum", "news", "blog"]
SENTIMENTS = ["positive", "negative", "neutral"]

st.set_page_config(page_title="Ask the Brand — Acme Co.", page_icon="🛼", layout="wide")
st.title("🛼 Ask the Brand — Acme Co. RAG")


# --------------------------------------------------------------------------- #
# Sidebar: connection, filters, retrieval knobs
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Connection")
    api_base = st.text_input("API base URL", value=DEFAULT_API)
    try:
        h = requests.get(f"{api_base}/health", timeout=5).json()
        st.success(f"Weaviate ready · {h['object_count']} chunks indexed")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API not reachable: {exc}")

    st.header("Filters")
    sel_sources = st.multiselect("Source", SOURCES)
    sel_sentiments = st.multiselect("Sentiment", SENTIMENTS)
    use_dates = st.checkbox("Filter by date range")
    date_from = date_to = None
    if use_dates:
        c1, c2 = st.columns(2)
        date_from = c1.date_input("From").isoformat()
        date_to = c2.date_input("To").isoformat()

    st.header("Retrieval")
    top_k = st.slider("top_k", 1, 20, 5)
    alpha = st.slider("alpha (0 = keyword/BM25, 1 = vector)", 0.0, 1.0, 0.5, 0.05)


def build_filters() -> dict:
    f = {}
    if sel_sources:
        f["sources"] = sel_sources
    if sel_sentiments:
        f["sentiments"] = sel_sentiments
    if use_dates:
        f["date_from"] = date_from
        f["date_to"] = date_to
    return f


def post(path: str, payload: dict) -> dict:
    r = requests.post(f"{api_base}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def render_citations(citations: list) -> None:
    if not citations:
        return
    st.markdown("**Citations**")
    for c in citations:
        st.markdown(
            f"- [`{c['doc_id']}`]({c['url']}) · {c['source']} · {c['posted_at'][:10]} · "
            f"{c['sentiment']} · score={c['score']:.3f}"
        )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
ask_tab, search_tab, sum_tab, cmp_tab, ingest_tab = st.tabs(
    ["Ask", "Search", "Summarize", "Compare", "Ingest"]
)

with ask_tab:
    st.subheader("Grounded Q&A")
    q = st.text_input("Question", "What do people say about the Rocket Skates battery life?")
    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving + answering..."):
            res = post("/ask", {"question": q, "filters": build_filters(),
                                "top_k": top_k, "alpha": alpha})
        if res["refused"]:
            st.warning(res["answer"])
        else:
            st.markdown(res["answer"])
        render_citations(res["citations"])

with search_tab:
    st.subheader("Raw hybrid retrieval (no LLM)")
    st.caption("Toggle filters and alpha in the sidebar to see retrieval change.")
    sq = st.text_input("Search query", "battery life")
    if st.button("Search"):
        res = post("/search", {"query": sq, "filters": build_filters(),
                               "top_k": top_k, "alpha": alpha})
        st.caption(f"alpha={res['alpha']} · top_k={res['top_k']} · filters={res['filters']}")
        for r in res["results"]:
            with st.expander(f"[{r['doc_id']}] {r['source']} · {r['sentiment']} · "
                             f"{r['posted_at'][:10]} · score={r['score']:.3f}"):
                st.write(r["body"])
                st.caption(r["url"])

with sum_tab:
    st.subheader("Summarize over a filter")
    instr = st.text_input("Instruction",
                          "Summarize the negative news about Acme from the last 30 days.")
    st.caption("Tip: set Source=news and Sentiment=negative in the sidebar.")
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            res = post("/summarize", {"instruction": instr, "filters": build_filters(),
                                      "top_k": max(top_k, 20)})
        st.caption(f"Grounded in {res['n_docs']} documents.")
        st.markdown(res["summary"])
        render_citations(res["citations"])

with cmp_tab:
    st.subheader("Before / after comparison")
    topic = st.text_input("Topic", "the Acme Rocket Skates")
    split = st.text_input("Split date (ISO)", "2026-04-10")
    if st.button("Compare"):
        with st.spinner("Comparing..."):
            res = post("/compare", {"topic": topic, "split_date": split,
                                    "filters": build_filters(), "top_k": max(top_k, 30)})
        st.caption(f"before={res['before_count']} docs · after={res['after_count']} docs · "
                   f"split={res['split_date']}")
        st.markdown(res["comparison"])
        render_citations(res["citations"])

with ingest_tab:
    st.subheader("Incremental ingestion")
    path = st.text_input("JSONL path", "data/new_docs.jsonl")
    if st.button("Ingest"):
        with st.spinner("Ingesting..."):
            res = post("/ingest", {"path": path})
        st.json(res)
        if res["new"] == 0 and res["changed"] == 0:
            st.info("Nothing new to embed — incremental ingestion working as intended.")
