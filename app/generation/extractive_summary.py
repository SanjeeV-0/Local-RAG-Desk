"""Local, deterministic extractive fallback summary.

Used ONLY when the primary generation call has already failed (see
`app.api.research._execute_research`) and retrieval already found usable,
PII-redacted sources. This module makes no network/provider call of any
kind -- it never retries the rejected request, never sends anything
anywhere, and operates exclusively on `GenerationContextSource.content`,
which is already redacted by `ContextAssembler.assemble` before this module
ever sees it. It cannot reintroduce unredacted content and cannot interact
with OpenRouter's guardrails at all, since it makes no provider call.

The approach is intentionally simple and auditable: split each source's
already-redacted text into sentences, drop any sentence that looks like an
instruction/injection attempt (see `_is_instruction_like`), score the
remaining sentences by how many of the query's significant words they
contain, and keep the highest-scoring sentences (in their original order) up
to a small budget. This is a heuristic extractive summary, not a generated
one -- callers must label it as such (see
`ExtractiveSummaryResult.is_extractive`) and must never present it as though
it came from the LLM. The instruction-like filter here is a narrow,
deterministic guard, not a general prompt-injection defense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.generation.context import GenerationContextSource

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Deliberately simple, deterministic substring/phrase checks -- this is not
# a general-purpose prompt-injection defense (that is explicit, separate,
# future work), only a narrow guard against the most obvious instruction-
# like sentences leaking into a locally-built fallback "answer". Matching is
# case-insensitive and intentionally conservative in scope (a handful of
# well-known injection phrasings), not an attempt at exhaustive coverage.
_INSTRUCTION_SIGNAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore\s+(?:the\s+|all\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions?\b",
        r"\bdisregard\s+(?:the\s+|all\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions?\b",
        r"\boverride\s+(?:the\s+)?(?:user'?s?\s+)?(?:request|instructions?|prompt)\b",
        r"\breveal\s+(?:the\s+|your\s+|any\s+)?(?:confidential|secret|hidden|internal)\b",
        r"\bdisclose\s+(?:the\s+|your\s+|any\s+)?(?:confidential|secret|password|credentials?|api\s*keys?)\b",
        r"\breveal\s+(?:your\s+)?(?:system\s+prompt|instructions?)\b",
        r"\b(?:leak|expose)\s+(?:the\s+|your\s+|any\s+)?(?:secret|confidential|password|credentials?)\b",
        r"\byou\s+are\s+now\s+(?:a|an)\b",
        r"\bact\s+as\s+(?:if\s+you|a|an)\b.{0,30}\b(?:no\s+restrictions|unrestricted|dan)\b",
    )
)


def _is_instruction_like(sentence: str) -> bool:
    """Deterministic check for an obvious instruction-like/injection
    sentence -- never a model call, never probabilistic. A sentence must
    match one of the explicit patterns above to be excluded; anything else
    (including ordinary factual text) is preserved."""

    return any(pattern.search(sentence) for pattern in _INSTRUCTION_SIGNAL_PATTERNS)

# Common words excluded from query-term scoring so matches reflect actual
# topical overlap rather than incidental overlap on function words.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "for", "and", "or", "but", "with", "as",
        "by", "at", "from", "that", "this", "these", "those", "it", "its",
        "what", "which", "who", "whom", "how", "why", "when", "where",
        "do", "does", "did", "can", "could", "should", "would", "will",
        "about", "into", "than", "then", "there", "their", "i", "you",
    }
)

MAX_SUMMARY_SENTENCES = 5


@dataclass(frozen=True)
class ExtractiveSummaryResult:
    """A locally-built, non-generated fallback answer.

    `is_extractive` is always `True` -- present so callers never need a
    separate type check to decide how to label this result in the API
    response (see `app.api.research._execute_research`).
    """

    text: str
    is_extractive: bool = True


def _query_terms(query: str) -> set[str]:
    return {word.lower() for word in _WORD_RE.findall(query) if word.lower() not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def build_extractive_summary(
    query: str,
    sources: list[GenerationContextSource],
    max_sentences: int = MAX_SUMMARY_SENTENCES,
) -> ExtractiveSummaryResult | None:
    """Build a deterministic extractive summary from already-redacted source
    text, or return `None` when there is no usable sentence to extract (for
    example, every source's redacted content was empty/whitespace-only, or
    every sentence present was instruction-like and excluded) -- callers
    must treat `None` as "safe summarization is not possible" and fall back
    to a plain safe error, never inventing text to fill the gap.

    Obvious instruction-like sentences (e.g. "ignore previous instructions",
    "reveal the confidential file") are excluded before scoring -- this
    operates purely on `source.content`, which is already PII-redacted by
    `ContextAssembler.assemble` before this function ever sees it; safe
    factual sentences from the same source are preserved and still eligible.
    """

    query_terms = _query_terms(query)

    candidates: list[tuple[int, int, str, int]] = []
    # (score, -source_index, sentence, source_index) -- source_index kept
    # for the attribution label; sort key below orders by score desc, then
    # by original source/sentence order for a fully deterministic result.

    for source_index, source in enumerate(sources, start=1):
        for sentence_index, sentence in enumerate(_split_sentences(source.content)):
            if _is_instruction_like(sentence):
                continue

            sentence_terms = _query_terms(sentence)
            score = len(query_terms & sentence_terms)
            candidates.append((score, source_index, sentence_index, sentence))

    if not candidates:
        return None

    # Highest score first; ties broken by first appearance (source order,
    # then sentence order within that source) -- fully deterministic, no
    # randomness and no dependence on dict/set iteration order for the
    # final ordering.
    ranked = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))
    top = ranked[:max_sentences]

    # Restore original (source, sentence) order for readability rather than
    # score order, once the top set has been chosen.
    top_sorted = sorted(top, key=lambda c: (c[1], c[2]))

    lines = [f"[Source {source_index}] {sentence}" for _, source_index, _, sentence in top_sorted]

    if not lines:
        return None

    return ExtractiveSummaryResult(text="\n".join(lines))
