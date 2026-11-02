"""Scoring functions. Each one takes plain values so it can be unit tested on its own."""

import re


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(any(r in relevant for r in retrieved[:k]))


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(retrieved[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    return next((1 / rank for rank, r in enumerate(retrieved, 1) if r in relevant), 0.0)


def fact_recall(answer: str, must_include: list[str]) -> float:
    """Share of expected facts present in the answer. "a|b" accepts either spelling.

    Numbers match as whole numbers, so "8" does not match "18".
    """
    if not must_include:
        return 1.0

    def present(fact: str) -> bool:
        return any(
            re.search(rf"(?<![\w.]){re.escape(alt.strip())}(?![\w])", answer, re.IGNORECASE)
            for alt in fact.split("|")
        )

    return sum(present(f) for f in must_include) / len(must_include)


def citation_precision(cited: list[str], evidence: list[str]) -> float:
    """Share of cited ids that were actually in the evidence shown to the model."""
    return sum(c in evidence for c in cited) / len(cited) if cited else 0.0


def cites_relevant(cited: list[str], relevant: set[str]) -> float:
    return float(any(c in relevant for c in cited))


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None
