"""Dependency-free, rule-based PII detection and redaction.

This module makes NO claim to comprehensive PII coverage. It detects exactly
four structurally-recognizable categories using deterministic rules -- there
is no confidence score and no probabilistic judgment anywhere here, by
design (an arbitrary confidence threshold would not be defensible for simple
regex/checksum rules, so every match is redacted unconditionally):

- Email addresses (pragmatic regex, not full RFC 5322).
- Phone numbers: North American formats plus a loose international `+<cc>`
  form. International coverage is intentionally limited -- many non-US/CA
  formats will NOT be detected. This is a known, documented gap, not an
  oversight.
- SSN-shaped numbers: strictly `###-##-####` (dashed). A bare 9-digit run is
  deliberately NOT matched -- it would be far too false-positive-prone
  against arbitrary numeric identifiers in research documents.
- Credit-card-shaped numbers: a 13-19 digit run (optionally grouped with
  spaces/dashes) that passes the Luhn checksum. The Luhn check is what keeps
  this precise -- it is what separates a real card-like number from an
  arbitrary long digit run (invoice numbers, ISBNs, etc.).

Names, physical addresses, account identifiers, and other context-dependent
sensitive information are explicitly OUT OF SCOPE for this rule-based
detector and would require separate, future work (e.g. an NLP-based
detector), not something this module should be assumed to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PiiCategory = Literal["email", "phone", "ssn", "credit_card"]

# Longer/more-specific categories win when spans overlap.
_CATEGORY_PRIORITY: dict[PiiCategory, int] = {
    "credit_card": 3,
    "ssn": 2,
    "phone": 1,
    "email": 0,
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# NANP-style phone numbers: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, and a
# loose `+<country><digits>` international form. Deliberately narrow -- many
# valid international formats will not match (documented limitation above).
_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{3}\)[\s-]?|\d{3}[\s.-])\d{3}[\s.-]\d{4}\b"
)

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Candidate digit runs for credit-card detection: 13-19 digits, optionally
# grouped with single spaces or dashes between groups. Luhn validation below
# is what actually decides whether a candidate is redacted.
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def _luhn_is_valid(digits: str) -> bool:
    """Standard Luhn checksum over a digit-only string."""

    total = 0
    reverse_digits = digits[::-1]

    for index, ch in enumerate(reverse_digits):
        digit = int(ch)

        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9

        total += digit

    return total % 10 == 0


@dataclass(frozen=True)
class PiiMatch:
    """A single detected PII span, in the ORIGINAL text's character offsets."""

    category: PiiCategory
    start: int
    end: int


@dataclass(frozen=True)
class RedactionResult:
    """Outcome of redacting a piece of text."""

    text: str
    matches: tuple[PiiMatch, ...]
    categories_found: frozenset[PiiCategory]

    @property
    def redacted(self) -> bool:
        return len(self.matches) > 0


_PLACEHOLDERS: dict[PiiCategory, str] = {
    "email": "[REDACTED:EMAIL]",
    "phone": "[REDACTED:PHONE]",
    "ssn": "[REDACTED:SSN]",
    "credit_card": "[REDACTED:CREDIT_CARD]",
}


def _raw_matches(text: str) -> list[PiiMatch]:
    """Collect every raw regex/checksum match, before overlap resolution."""

    found: list[PiiMatch] = []

    for match in _EMAIL_RE.finditer(text):
        found.append(PiiMatch("email", match.start(), match.end()))

    for match in _PHONE_RE.finditer(text):
        found.append(PiiMatch("phone", match.start(), match.end()))

    for match in _SSN_RE.finditer(text):
        found.append(PiiMatch("ssn", match.start(), match.end()))

    for match in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"[ -]", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_is_valid(digits):
            found.append(PiiMatch("credit_card", match.start(), match.end()))

    return found


def _resolve_overlaps(matches: list[PiiMatch]) -> list[PiiMatch]:
    """Drop fully/partially overlapping matches, keeping the longer span
    (ties broken by category priority: credit_card > ssn > phone > email).
    The discarded span is dropped entirely -- spans are never merged."""

    ordered = sorted(
        matches,
        key=lambda m: (m.start, -(m.end - m.start), -_CATEGORY_PRIORITY[m.category]),
    )

    resolved: list[PiiMatch] = []
    last_end = -1

    for match in ordered:
        if match.start < last_end:
            # Overlaps the previously kept match -- the previous one was
            # chosen first because it starts no later and (via the sort
            # key) is at least as long/high-priority. Discard this one.
            continue

        resolved.append(match)
        last_end = match.end

    return resolved


def detect(text: str) -> list[PiiMatch]:
    """Return non-overlapping PII matches in `text`, in left-to-right order."""

    return _resolve_overlaps(_raw_matches(text))


def redact(text: str) -> RedactionResult:
    """Redact every detected PII span in `text` with a category placeholder.

    Text outside matched spans is preserved exactly -- a chunk is never
    discarded wholesale, only the specific sensitive spans within it.
    """

    matches = detect(text)

    if not matches:
        return RedactionResult(text=text, matches=(), categories_found=frozenset())

    pieces: list[str] = []
    cursor = 0

    for match in matches:
        pieces.append(text[cursor : match.start])
        pieces.append(_PLACEHOLDERS[match.category])
        cursor = match.end

    pieces.append(text[cursor:])

    return RedactionResult(
        text="".join(pieces),
        matches=tuple(matches),
        categories_found=frozenset(m.category for m in matches),
    )
