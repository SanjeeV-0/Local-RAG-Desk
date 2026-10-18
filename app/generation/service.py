"""Provider-agnostic generation contract. `app.generation.openrouter` is the
only implementation; kept as a `Protocol` so `GenerationService`/tests don't
depend on the OpenAI SDK directly.
"""

from dataclasses import dataclass
from typing import Protocol

from app.generation.context import GenerationContext


@dataclass(frozen=True)
class GenerationResult:
    """Result returned by a generation provider.

    `answer_redacted` is set by `GenerationService` (never by a
    `GenerationProvider` implementation itself) when the final generated
    `text` contained a PII-shaped span that had to be redacted after
    generation -- see `app.privacy.pii_detector`. It is always `False` for
    a result returned directly by a provider.
    """

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    answer_redacted: bool = False
    # The provider's own completion-stop reason (e.g. `"stop"`, `"length"`),
    # when exposed -- `"length"` means the response was cut off by
    # `max_tokens`, not that the model chose to stop; surfaced so an
    # incomplete answer can be diagnosed as a budget problem rather than a
    # content/quality one. `None` for a provider that doesn't expose it.
    finish_reason: str | None = None


class GenerationProvider(Protocol):
    """Interface implemented by an LLM generation provider."""

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """Generate an answer using the supplied context."""
