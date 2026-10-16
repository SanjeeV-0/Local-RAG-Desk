from app.generation.context import GenerationContext
from app.generation.errors import ProviderUnavailableError
from app.generation.service import (
    GenerationProvider,
    GenerationResult,
)
from app.observability.langfuse import get_langfuse
from app.privacy.pii_detector import redact


def _redact_result(result: GenerationResult) -> GenerationResult:
    """Final-answer PII scan: a defense-in-depth pass over the model's
    generated text, independent of (and run regardless of) whether Langfuse
    is enabled. This is NOT a substitute for the upstream context redaction
    already applied in `ContextAssembler.assemble` -- it exists only to
    catch the narrower case where a model reconstructs, paraphrases, or
    otherwise reproduces a PII-shaped pattern not caught upstream. A simple
    regex/checksum pass cannot reliably detect every paraphrased or
    contextual disclosure -- this is a documented limitation, not a claim of
    completeness.
    """

    answer_redaction = redact(result.text)

    if not answer_redaction.redacted:
        return result

    return GenerationResult(
        text=answer_redaction.text,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        answer_redacted=True,
    )


class GenerationService:
    """Thin wrapper that adds a final-answer PII redaction pass and optional
    Langfuse observability around a `GenerationProvider` call.

    The redaction pass (`_redact_result`) always runs, regardless of whether
    Langfuse is configured -- observability can never change generation
    *behavior* beyond that, only observe it, preserving the existing
    "Langfuse never changes retrieval/generation behavior" invariant for
    everything else this service does.
    """

    def __init__(
        self,
        provider: GenerationProvider,
    ) -> None:
        self.provider = provider

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """Generate an answer using the supplied context."""

        langfuse = get_langfuse()

        if langfuse is None:
            result = await self.provider.generate(
                query=query,
                context=context,
            )
            return _redact_result(result)

        # Never send raw document context or raw query text to Langfuse --
        # only safe, non-content metadata. `query` is redacted here as a
        # defensive second pass (it is already redacted by the time it
        # reaches this service -- see `RetrievalService.search`); `context`
        # is never included at all, since `ContextAssembler` already
        # redacted the chunk content it's built from, and there is no safe
        # reason to also ship the full assembled text to a third party.
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="answer-generation",
            input={
                "query": redact(query).text,
                "context_length": len(context.text),
                "source_count": len(context.sources),
                "redacted_source_count": context.redacted_source_count,
            },
        ) as observation:
            try:
                result = await self.provider.generate(
                    query=query,
                    context=context,
                )

                result = _redact_result(result)

                observation.update(
                    output={
                        "text": result.text,
                        "answer_redacted": result.answer_redacted,
                        "model": result.model,
                        "usage": {
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "total_tokens": result.total_tokens,
                        },
                    },
                )

                return result

            except ProviderUnavailableError as exc:
                observation.update(
                    level="ERROR",
                    status_message=exc.category.value,
                )
                raise

            except Exception as exc:
                # Any other, unclassified exception -- still never send its
                # raw text to Langfuse.
                observation.update(
                    level="ERROR",
                    status_message=exc.__class__.__name__,
                )
                raise
