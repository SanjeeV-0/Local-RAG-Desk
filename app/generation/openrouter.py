"""The only `GenerationProvider` implementation: calls the answer-generation
LLM through OpenRouter's OpenAI-compatible API.

`temperature`/`max_tokens`/`top_p`/`response_format`/`reasoning_effort`/
`top_k`/`frequency_penalty`/`presence_penalty` are explicit, per-instance
configuration (sourced from `Settings.openrouter_generation_*` via
`app.core.dependencies.get_generation_provider`) rather than left to
whatever the selected OpenRouter model defaults to. `max_tokens`/
`response_format`/`reasoning_effort`/`top_k`/`frequency_penalty`/
`presence_penalty` are optional -- only included in the actual request when
configured (not `None`); `temperature`/`top_p` are always sent explicitly.

`top_k` is NOT part of the OpenAI chat-completion API -- it is an
OpenRouter-specific passthrough field, sent via `extra_body` rather than as
a direct keyword argument (see `generate` below), and only an upstream
model that actually supports it will honor it; this is a deliberate
exception to "send every configured field as a plain kwarg".

No custom retry loop is implemented -- `max_retries` is passed straight
through to `AsyncOpenAI`, which already implements its own retry/backoff
internally for retryable errors.
"""

from openai import AsyncOpenAI

from app.generation.context import GenerationContext
from app.generation.service import GenerationResult


class OpenRouterGenerationProvider:
    """Generate answers through the OpenRouter OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int | None,
        top_p: float,
        response_format: dict[str, object] | None,
        reasoning_effort: str | None,
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "Production AI Research & Knowledge Agent",
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.response_format = response_format
        self.reasoning_effort = reasoning_effort
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # `timeout` is only passed to the client when actually configured --
        # the `openai` SDK distinguishes "not given at all" (its own
        # library-internal default applies) from an explicit value, so an
        # unconfigured (`None`) setting here must not become an explicit
        # `timeout=None` override of that distinction.
        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "base_url": base_url,
            "default_headers": {
                "X-Title": app_name,
            },
            "max_retries": max_retries,
        }

        if timeout is not None:
            client_kwargs["timeout"] = timeout

        self.client = AsyncOpenAI(**client_kwargs)

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """Generate an answer using OpenRouter."""

        # `temperature`/`top_p` are always explicit, configured values.
        # `max_tokens`/`response_format`/`reasoning_effort`/
        # `frequency_penalty`/`presence_penalty` are only included when
        # actually configured (not None), so an unconfigured field never
        # reaches the API as an explicit override.
        request_kwargs: dict[str, object] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. "
                        "Answer the user's question using the "
                        "provided research context. "
                        "Do not invent facts that are not supported "
                        "by the context."
                    ),
                },
                {
                    "role": "user",
                    "content": (f"Research context:\n\n{context.text}\n\nQuestion:\n\n{query}"),
                },
            ],
        }

        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        if self.response_format is not None:
            request_kwargs["response_format"] = self.response_format

        if self.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.reasoning_effort

        if self.frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = self.frequency_penalty

        if self.presence_penalty is not None:
            request_kwargs["presence_penalty"] = self.presence_penalty

        if self.top_k is not None:
            # OpenRouter-specific passthrough, not a standard OpenAI field --
            # must travel via extra_body, never as a direct kwarg (the
            # openai SDK's typed `.create()` has no `top_k` parameter at
            # all).
            request_kwargs["extra_body"] = {"top_k": self.top_k}

        response = await self.client.chat.completions.create(**request_kwargs)

        message = response.choices[0].message.content

        if message is None:
            raise RuntimeError("OpenRouter returned an empty response.")

        usage = response.usage

        return GenerationResult(
            text=message,
            model=response.model or self.model,
            input_tokens=(usage.prompt_tokens if usage is not None else None),
            output_tokens=(usage.completion_tokens if usage is not None else None),
            total_tokens=(usage.total_tokens if usage is not None else None),
        )
