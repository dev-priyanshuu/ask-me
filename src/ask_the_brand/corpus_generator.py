"""Generate a synthetic corpus of ~1,500 short documents about the fictional brand "Acme Co.".

Design
------
* A *deterministic* plan (seeded) decides, for every document, its theme, source,
  author, sentiment and posted_at date. This guarantees a reproducible mix and a
  realistic sentiment/source/date distribution.
* The document **body** is the only field written by the LLM (batched, cheap). With
  ``--offline`` (or when no OPENAI_API_KEY is set) bodies are filled from templates so
  the whole pipeline is runnable without spending tokens.
* A handful of **anchor** docs (known facts) and **contradiction pairs** are hardcoded
  with stable ids so the eval set has reliable ground truth and the "contradictory
  items" requirement is guaranteed.

Each document has: id, source, posted_at, author, body, url, sentiment.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings

SEED = 42
BASE_URL = "https://acme.example.com"

SOURCES_SOCIAL = ["twitter", "reddit", "forum"]
SOURCES_NEWS = ["news", "blog"]
SENTIMENTS = ["positive", "negative", "neutral"]

AUTHORS = {
    "twitter": ["@skatefanatic", "@gadgetgwen", "@dailydose_tech", "@marco_rolls", "@nina_reviews",
                "@urbanwheeler", "@techskeptic", "@happycustomer22", "@rollwithit", "@acme_hater"],
    "reddit": ["u/quietcoder", "u/pnw_skater", "u/budget_buyer", "u/r_gadgets_mod", "u/longtime_lurker",
               "u/throwaway9981", "u/diy_dan", "u/skeptic_sam", "u/early_adopter", "u/refund_please"],
    "forum": ["wheelhead", "RollModel", "voltage_vic", "CautiousCarol", "FastEddie", "GearGuru",
              "newbie_nate", "ProSkaterPaula"],
    "news": ["The Daily Ledger", "TechWire", "Market Beat", "Consumer Watch", "The Verge Times",
             "Reuters-ish Wire", "City Tribune"],
    "blog": ["Gadget Diaries", "The Skate Blog", "EcoTech Notes", "Startup Pulse", "Review Lab"],
}

# Theme -> configuration. `keywords` seed the LLM/template; `date_window` biases when
# docs about a theme were posted; `sentiment_bias` skews the sentiment mix.
THEMES: Dict[str, dict] = {
    "rocket_skates_launch": {
        "label": "the launch of Acme Rocket Skates (a motorized personal skate, launched 2026-04-10)",
        "angles": [
            "first impressions after trying the Rocket Skates",
            "the pre-order experience and shipping wait",
            "how the Rocket Skates compare to a competitor's product",
            "the launch event livestream",
            "build quality and design of the Rocket Skates",
            "whether the Rocket Skates are worth the price",
        ],
        "date_window": ("2026-03-20", "2026-05-25"),
        "sentiment_bias": [0.4, 0.35, 0.25],
    },
    "customer_support": {
        "label": "Acme Co. customer support and warranty experiences",
        "angles": [
            "a long wait time with support",
            "a surprisingly helpful support agent",
            "a warranty claim that was approved",
            "a warranty claim that was denied",
            "trouble reaching anyone by phone",
            "a chatbot that didn't help",
        ],
        "date_window": ("2026-01-01", "2026-05-31"),
        "sentiment_bias": [0.3, 0.45, 0.25],
    },
    "pricing": {
        "label": "Acme Prime, the $9.99/month subscription, and recent pricing changes",
        "angles": [
            "the value of the Acme Prime subscription",
            "a surprise price increase",
            "comparing Acme prices to rivals",
            "a discount or promo code",
            "whether the subscription is worth keeping",
        ],
        "date_window": ("2026-01-01", "2026-05-31"),
        "sentiment_bias": [0.3, 0.4, 0.3],
    },
    "sustainability": {
        "label": "Acme Co.'s sustainability pledge to use 100% recycled packaging by 2027",
        "angles": [
            "praise for the recycled packaging",
            "skepticism about greenwashing",
            "the carbon-neutral shipping claim",
            "the trade-in and recycling program",
            "the annual sustainability report",
        ],
        "date_window": ("2026-01-01", "2026-05-31"),
        "sentiment_bias": [0.45, 0.25, 0.3],
    },
    "recall": {
        "label": "a safety controversy over Acme Rocket Skates overheating",
        "angles": [
            "reports of a skate overheating during use",
            "calls for a recall",
            "Acme's official statement on safety",
            "a regulator looking into the reports",
            "advice to stop using the skates until fixed",
        ],
        "date_window": ("2026-04-25", "2026-05-31"),
        "sentiment_bias": [0.1, 0.7, 0.2],
    },
    "battery": {
        "label": "the battery life and performance of Acme Rocket Skates",
        "angles": [
            "how long the battery lasts on a charge",
            "charging speed",
            "performance on hills",
            "battery degradation over weeks of use",
            "top speed and range",
        ],
        "date_window": ("2026-04-05", "2026-05-31"),
        "sentiment_bias": [0.35, 0.4, 0.25],
    },
}


@dataclass
class Doc:
    id: str
    source: str
    posted_at: str
    author: str
    body: str
    url: str
    sentiment: str


@dataclass
class _Spec:
    """A planned document whose body has not been written yet."""

    id: str
    source: str
    posted_at: str
    author: str
    url: str
    sentiment: str
    theme: str
    angle: str


# --------------------------------------------------------------------------- #
# Hardcoded anchor facts (reliable ground truth for eval)
# --------------------------------------------------------------------------- #
ANCHOR_DOCS: List[Doc] = [
    Doc(
        id="acme-anchor-launch",
        source="news",
        posted_at="2026-04-10T09:00:00Z",
        author="TechWire",
        body=("Acme Co. today launched the Acme Rocket Skates, its first motorized personal "
              "skate, at a price of $249. The company says the skates began shipping to "
              "pre-order customers on April 10, 2026."),
        url=f"{BASE_URL}/news/acme-anchor-launch",
        sentiment="neutral",
    ),
    Doc(
        id="acme-anchor-ceo",
        source="news",
        posted_at="2026-02-14T11:30:00Z",
        author="The Daily Ledger",
        body=("In an interview, Acme Co. chief executive Dana Whitfield outlined the company's "
              "roadmap for 2026, calling the upcoming Rocket Skates the brand's most important "
              "launch in a decade."),
        url=f"{BASE_URL}/news/acme-anchor-ceo",
        sentiment="neutral",
    ),
    Doc(
        id="acme-anchor-prime-price",
        source="blog",
        posted_at="2026-01-20T15:00:00Z",
        author="Review Lab",
        body=("Acme Prime, the company's membership, costs $9.99 per month and bundles free "
              "two-day shipping, an extended warranty, and members-only discounts."),
        url=f"{BASE_URL}/blog/acme-anchor-prime-price",
        sentiment="neutral",
    ),
    Doc(
        id="acme-anchor-sustainability",
        source="news",
        posted_at="2026-03-05T08:00:00Z",
        author="Consumer Watch",
        body=("Acme Co. pledged to move to 100% recycled packaging by 2027 and to offer "
              "carbon-neutral shipping on all orders as part of its new sustainability program."),
        url=f"{BASE_URL}/news/acme-anchor-sustainability",
        sentiment="positive",
    ),
    Doc(
        id="acme-anchor-support",
        source="blog",
        posted_at="2026-01-10T12:00:00Z",
        author="Gadget Diaries",
        body=("Acme Co. customer support is available from 9am to 6pm Pacific, Monday through "
              "Friday, by phone, email, and live chat. Acme Prime members get priority support."),
        url=f"{BASE_URL}/blog/acme-anchor-support",
        sentiment="neutral",
    ),
]

# --------------------------------------------------------------------------- #
# Hardcoded contradiction pairs (guaranteed contradictory items)
# --------------------------------------------------------------------------- #
CONTRADICTION_DOCS: List[Doc] = [
    # Battery life: 20 hours vs dies in 3 hours
    Doc("acme-contra-battery-20h", "news", "2026-04-12T10:00:00Z", "TechWire",
        "Acme says the Rocket Skates deliver up to 20 hours of battery life on a single "
        "charge, easily covering a week of commutes.",
        f"{BASE_URL}/news/acme-contra-battery-20h", "positive"),
    Doc("acme-contra-battery-3h", "twitter", "2026-04-15T18:20:00Z", "@techskeptic",
        "Whatever Acme claims, my Rocket Skates battery dies in about 3 hours of real riding. "
        "Nowhere near the advertised number. Hugely disappointing.",
        f"{BASE_URL}/twitter/acme-contra-battery-3h", "negative"),
    # Recall: official recall vs Acme denies any recall
    Doc("acme-contra-recall-yes", "news", "2026-05-08T09:30:00Z", "Consumer Watch",
        "Acme Co. has issued a voluntary recall of the Rocket Skates after multiple reports "
        "of batteries overheating, advising owners to stop using them immediately.",
        f"{BASE_URL}/news/acme-contra-recall-yes", "negative"),
    Doc("acme-contra-recall-no", "news", "2026-05-08T16:45:00Z", "Market Beat",
        "An Acme Co. spokesperson denied that any recall of the Rocket Skates has been issued, "
        "calling the overheating reports isolated and saying the product remains on sale.",
        f"{BASE_URL}/news/acme-contra-recall-no", "neutral"),
    # Pricing: doubled vs unchanged
    Doc("acme-contra-price-up", "reddit", "2026-03-02T13:00:00Z", "u/budget_buyer",
        "Just noticed Acme Prime doubled in price overnight from $9.99 to $19.99. Cancelling "
        "today, this is ridiculous.",
        f"{BASE_URL}/reddit/acme-contra-price-up", "negative"),
    Doc("acme-contra-price-same", "blog", "2026-03-03T10:00:00Z", "Startup Pulse",
        "Despite the online rumors, Acme confirmed that the Acme Prime subscription price is "
        "unchanged at $9.99 per month and that no increase is planned.",
        f"{BASE_URL}/blog/acme-contra-price-same", "neutral"),
]


# --------------------------------------------------------------------------- #
# Planning (deterministic)
# --------------------------------------------------------------------------- #
def _rand_date(rng: random.Random, start: str, end: str) -> str:
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    delta = int((e - s).total_seconds())
    dt = s + timedelta(seconds=rng.randint(0, delta))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _pick_sentiment(rng: random.Random, bias: List[float]) -> str:
    return rng.choices(SENTIMENTS, weights=bias, k=1)[0]


def build_plan(n_docs: int, rng: random.Random, id_prefix: str = "acme",
               date_override: Optional[tuple] = None) -> List[_Spec]:
    """Create a deterministic list of doc specs (no bodies yet)."""
    themes = list(THEMES.keys())
    # Weight the launch / battery / recall themes a bit higher so the corpus has
    # strong recurring threads to retrieve and compare.
    theme_weights = {
        "rocket_skates_launch": 0.30,
        "customer_support": 0.16,
        "pricing": 0.14,
        "sustainability": 0.12,
        "recall": 0.13,
        "battery": 0.15,
    }
    weights = [theme_weights[t] for t in themes]

    specs: List[_Spec] = []
    for i in range(n_docs):
        theme = rng.choices(themes, weights=weights, k=1)[0]
        cfg = THEMES[theme]
        # ~60% social, 40% news-style.
        if rng.random() < 0.6:
            source = rng.choice(SOURCES_SOCIAL)
        else:
            source = rng.choice(SOURCES_NEWS)
        author = rng.choice(AUTHORS[source])
        sentiment = _pick_sentiment(rng, cfg["sentiment_bias"])
        window = date_override or cfg["date_window"]
        posted_at = _rand_date(rng, window[0], window[1])
        angle = rng.choice(cfg["angles"])
        doc_id = f"{id_prefix}-{i+1:04d}"
        specs.append(_Spec(
            id=doc_id,
            source=source,
            posted_at=posted_at,
            author=author,
            url=f"{BASE_URL}/{source}/{doc_id}",
            sentiment=sentiment,
            theme=theme,
            angle=angle,
        ))
    return specs


# --------------------------------------------------------------------------- #
# Body generation
# --------------------------------------------------------------------------- #
_STYLE = {
    "twitter": "a casual tweet, 1-2 sentences, under 240 characters, may use 1 hashtag",
    "reddit": "a short Reddit comment, 1-3 sentences, conversational",
    "forum": "a brief forum post, 1-3 sentences",
    "news": "a 2-3 sentence news snippet, neutral journalistic tone, third person",
    "blog": "a 2-3 sentence blog excerpt, slightly more personal than news",
}


def _offline_body(spec: _Spec) -> str:
    """Template fallback used when no LLM is available."""
    cfg = THEMES[spec.theme]
    tone = {
        "positive": "I'm really impressed —",
        "negative": "Honestly disappointed:",
        "neutral": "For the record,",
    }[spec.sentiment]
    return (f"{tone} on {spec.angle} regarding {cfg['label']}. "
            f"({spec.source} take from {spec.author}.)")


def _build_batch_prompt(specs: List[_Spec]) -> str:
    lines = []
    for idx, s in enumerate(specs):
        cfg = THEMES[s.theme]
        lines.append(
            f"{idx}. source={s.source} | style={_STYLE[s.source]} | sentiment={s.sentiment} | "
            f"topic={cfg['label']} | angle={s.angle}"
        )
    specs_block = "\n".join(lines)
    return (
        "You are generating a synthetic corpus about a FICTIONAL brand named \"Acme Co.\" "
        "Write the BODY text for each item below. Mention \"Acme\" or the relevant product "
        "naturally. Match the requested source style and sentiment exactly. Vary wording so "
        "items don't sound templated. Do not invent real companies or real people. Do not add "
        "quotation marks around the whole body.\n\n"
        f"ITEMS:\n{specs_block}\n\n"
        "Return ONLY a JSON object of the form {\"items\": [{\"i\": <index>, \"body\": <text>}, ...]} "
        "with exactly one entry per item index above."
    )


def _generate_bodies_llm(specs: List[_Spec], batch_size: int = 20) -> Dict[int, str]:
    """Fill bodies via the LLM in batches. Returns {plan_index: body}."""
    from langchain_openai import ChatOpenAI
    from tenacity import retry, stop_after_attempt, wait_exponential

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0.9,
        model_kwargs={"response_format": {"type": "json_object"}},
        api_key=settings.openai_api_key,
    )

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _call(batch: List[_Spec]) -> Dict[int, str]:
        resp = llm.invoke(_build_batch_prompt(batch))
        data = json.loads(resp.content)
        out = {}
        for item in data["items"]:
            out[int(item["i"])] = str(item["body"]).strip()
        return out

    try:
        from tqdm import tqdm
        batches = list(range(0, len(specs), batch_size))
        iterator = tqdm(batches, desc="Generating bodies", unit="batch")
    except ImportError:  # pragma: no cover
        iterator = range(0, len(specs), batch_size)

    bodies: Dict[int, str] = {}
    for start in iterator:
        batch = specs[start:start + batch_size]
        local = _call(batch)
        for local_idx, body in local.items():
            if 0 <= local_idx < len(batch):
                bodies[start + local_idx] = body
    # Fill any gaps with the offline template so we never emit empty bodies.
    for i, s in enumerate(specs):
        bodies.setdefault(i, _offline_body(s))
    return bodies


def specs_to_docs(specs: List[_Spec], offline: bool) -> List[Doc]:
    if offline:
        bodies = {i: _offline_body(s) for i, s in enumerate(specs)}
    else:
        bodies = _generate_bodies_llm(specs)
    docs = []
    for i, s in enumerate(specs):
        docs.append(Doc(
            id=s.id, source=s.source, posted_at=s.posted_at, author=s.author,
            body=bodies[i], url=s.url, sentiment=s.sentiment,
        ))
    return docs


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _write_jsonl(docs: List[Doc], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def generate_corpus(out_path: str, n_docs: int = 1500, offline: bool = False,
                    new_docs_path: Optional[str] = None, n_new: int = 25) -> Dict[str, int]:
    """Generate the main corpus (+ optional incremental new_docs file)."""
    rng = random.Random(SEED)

    # Reserve room for the hardcoded anchors + contradictions so totals land ~n_docs.
    fixed = ANCHOR_DOCS + CONTRADICTION_DOCS
    n_generated = max(0, n_docs - len(fixed))

    specs = build_plan(n_generated, rng, id_prefix="acme")
    generated = specs_to_docs(specs, offline=offline)
    all_docs = generated + fixed
    # Stable order: by date.
    all_docs.sort(key=lambda d: d.posted_at)
    _write_jsonl(all_docs, Path(out_path))

    result = {"corpus": len(all_docs)}

    if new_docs_path:
        rng_new = random.Random(SEED + 1)
        new_specs = build_plan(n_new, rng_new, id_prefix="acme-new",
                               date_override=("2026-06-01", "2026-06-30"))
        new_docs = specs_to_docs(new_specs, offline=offline)
        # One known anchor in the new batch so the incremental demo has a checkable fact.
        new_docs.append(Doc(
            id="acme-new-anchor-v2",
            source="news",
            posted_at="2026-06-15T09:00:00Z",
            author="TechWire",
            body=("Acme Co. announced the Rocket Skates v2, a lighter model with a redesigned "
                  "battery the company says addresses earlier overheating complaints. It ships "
                  "in July 2026 at $279."),
            url=f"{BASE_URL}/news/acme-new-anchor-v2",
            sentiment="positive",
        ))
        new_docs.sort(key=lambda d: d.posted_at)
        _write_jsonl(new_docs, Path(new_docs_path))
        result["new_docs"] = len(new_docs)

    return result


def load_jsonl(path: str) -> List[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs
