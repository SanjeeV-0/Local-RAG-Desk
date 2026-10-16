from dataclasses import dataclass
from uuid import UUID

from app.privacy.pii_detector import redact
from app.retrieval.models import RetrievedChunk


@dataclass(frozen=True)
class GenerationContextSource:
    """Source metadata for a chunk used during generation.

    `content` is this source's PII-redacted text (identical to what was
    included in `GenerationContext.text`) -- kept here, per-source, so a
    local extractive fallback summary (see
    `app.generation.extractive_summary`) can be built from already-redacted
    text with attribution, without re-deriving it from the raw retrieval
    results or ever operating on unredacted content.
    """

    document_id: UUID
    chunk_id: UUID
    section_id: UUID
    section_path: str
    page_numbers: list[int]
    content: str


@dataclass(frozen=True)
class GenerationContext:
    """Context assembled for a future generation request.

    `redacted_source_count` is how many of `sources` had PII-shaped content
    redacted before inclusion -- surfaced to callers so a redaction can be
    reported (see `app.privacy.schemas.RedactionOutcome`) rather than
    silently applied.
    """

    text: str
    sources: list[GenerationContextSource]
    redacted_source_count: int = 0


class ContextAssembler:
    """Assemble retrieved chunks into generation context.

    Pure and deterministic: no LLM call, no I/O. This is the explicit
    boundary between "retrieval" (producing ranked `RetrievedChunk`s) and
    "generation" (turning them into a prompt and calling an LLM) -- the
    text/source format produced here is what both `RetrievalService`'s trace
    `context` and `/research/ask`'s actual generation call consume.

    Every chunk's content is passed through `app.privacy.pii_detector.redact`
    before being included -- this is the single shared chokepoint that
    guarantees redacted content reaches both the OpenRouter generation
    request and any trace/diagnostic view built from this context, so the
    two can never diverge (one showing raw text, the other redacted).

    Every real call site (`RetrievalService`, `app.api.research`) constructs
    this with no `max_characters`, so there is currently no context-length
    cap applied in practice -- `max_characters` exists but is unused by the
    application today.
    """

    def __init__(self, max_characters: int | None = None) -> None:
        if max_characters is not None and max_characters < 0:
            raise ValueError("max_characters must be greater than or equal to zero.")

        self.max_characters = max_characters

    def assemble(
        self,
        chunks: list[RetrievedChunk],
    ) -> GenerationContext:
        """Build generation context from retrieved chunks.

        Chunks are included whole or not at all -- a chunk is never
        truncated mid-text to fit `max_characters`; once adding the next
        chunk would exceed the budget, assembly stops.
        """

        text_parts: list[str] = []
        sources: list[GenerationContextSource] = []
        current_length = 0
        redacted_source_count = 0

        for index, chunk in enumerate(chunks, start=1):
            redaction = redact(chunk.content)

            if redaction.redacted:
                redacted_source_count += 1

            source_text = f"[Source {index}]\n{redaction.text}"

            additional_length = len(source_text)

            if text_parts:
                additional_length += 2

            if (
                self.max_characters is not None
                and current_length + additional_length > self.max_characters
            ):
                break

            text_parts.append(source_text)

            sources.append(
                GenerationContextSource(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    section_id=chunk.section_id,
                    section_path=chunk.section_path,
                    page_numbers=chunk.page_numbers,
                    content=redaction.text,
                )
            )

            current_length += additional_length

        return GenerationContext(
            text="\n\n".join(text_parts),
            sources=sources,
            redacted_source_count=redacted_source_count,
        )
