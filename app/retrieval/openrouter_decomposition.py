"""The only `QueryDecompositionProvider` implementation: calls an LLM
through OpenRouter's OpenAI-compatible API to split a query into focused
retrieval sub-queries. Raw/malformed output is this provider's problem to
raise on (ValueError/RuntimeError) -- validating and recovering from that is
`app.retrieval.query_decomposer.QueryDecomposer`'s job, not this file's.

No custom retry loop is implemented -- `max_retries` is passed straight
through to `AsyncOpenAI`, which already implements its own retry/backoff
internally for retryable errors.
"""

import json

from openai import AsyncOpenAI


class OpenRouterQueryDecompositionProvider:
    """Generate focused retrieval queries through OpenRouter."""

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

    async def decompose(self, query: str) -> list[str]:
        """Ask the model to split a multi-intent query into focused queries.

        Raises `RuntimeError`/`ValueError` for an empty response or any
        shape that doesn't match `{"queries": [str, ...]}` -- these are
        caught by `QueryDecomposer.decompose`, which falls back to the
        original query alone rather than propagating the failure.
        """

        # `temperature`/`top_p` are always explicit (Settings defaults
        # decomposition's temperature to 0.0 for deterministic, structural
        # query planning -- but that is now a configured value, not a
        # hard-coded one here). `max_tokens`/`response_format`/
        # `reasoning_effort`/`frequency_penalty`/`presence_penalty` are only
        # included when actually configured, so an unconfigured (None)
        # value never reaches the API as an explicit override of the
        # provider's own default behavior for that field.
        request_kwargs: dict[str, object] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a retrieval query planner. "
                        "Split the user's question only when it contains "
                        "multiple distinct information needs. "
                        "Generate focused retrieval questions for those needs. "
                        "Do not generate paraphrases, semantic variants, "
                        "or arbitrary query expansions. "
                        "Return at most five queries. "
                        'Return JSON only in the form {"queries": ["..."]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": query,
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
            raise RuntimeError("OpenRouter returned an empty decomposition response.")

        try:
            payload = json.loads(message)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenRouter returned invalid JSON for query decomposition.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Query decomposition response must be a JSON object.")

        queries = payload.get("queries")

        if not isinstance(queries, list):
            raise ValueError("Query decomposition response must contain a 'queries' list.")

        if not all(isinstance(query, str) for query in queries):
            raise ValueError("All decomposed queries must be strings.")

        return queries
