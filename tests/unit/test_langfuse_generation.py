import pytest

from app.generation.context import ContextAssembler
from app.generation.generation_service import GenerationService
from app.generation.service import GenerationResult


class FakeGenerationProvider:
    def __init__(self) -> None:
        self.received_query = None
        self.received_context = None

    async def generate(
        self,
        query: str,
        context,
    ) -> GenerationResult:
        self.received_query = query
        self.received_context = context

        return GenerationResult(
            text="Generated answer.",
            model="fake-model",
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
        )


class FakeObservation:
    def __init__(
        self,
        name: str,
        observation_type: str,
        input_data: dict,
    ) -> None:
        self.name = name
        self.observation_type = observation_type
        self.input_data = input_data
        self.output = None
        self.level = None
        self.status_message = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def update(
        self,
        *,
        output=None,
        level=None,
        status_message=None,
    ) -> None:
        if output is not None:
            self.output = output

        if level is not None:
            self.level = level

        if status_message is not None:
            self.status_message = status_message


class FakeLangfuse:
    def __init__(self) -> None:
        self.observations = []

    def start_as_current_observation(
        self,
        *,
        as_type,
        name,
        input,
    ):
        observation = FakeObservation(
            name=name,
            observation_type=as_type,
            input_data=input,
        )

        self.observations.append(observation)

        return observation


@pytest.mark.asyncio
async def test_generation_creates_langfuse_observation(
    monkeypatch,
) -> None:
    fake_langfuse = FakeLangfuse()
    provider = FakeGenerationProvider()

    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = GenerationService(provider)

    context = ContextAssembler().assemble(
        [],
    )

    result = await service.generate(
        query="What is RAG?",
        context=context,
    )

    assert result.text == "Generated answer."
    assert result.model == "fake-model"

    assert len(fake_langfuse.observations) == 1

    observation = fake_langfuse.observations[0]

    assert observation.name == "answer-generation"
    assert observation.observation_type == "generation"

    # Raw context is never sent to Langfuse -- only safe, non-content
    # metadata (see app.generation.generation_service.GenerationService).
    assert observation.input_data == {
        "query": "What is RAG?",
        "context_length": 0,
        "source_count": 0,
        "redacted_source_count": 0,
    }

    assert observation.output == {
        "text": "Generated answer.",
        "answer_redacted": False,
        "model": "fake-model",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
        },
    }


@pytest.mark.asyncio
async def test_generation_without_langfuse_still_works(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: None,
    )

    provider = FakeGenerationProvider()
    service = GenerationService(provider)

    context = ContextAssembler().assemble([])

    result = await service.generate(
        query="What is RAG?",
        context=context,
    )

    assert result.text == "Generated answer."
    assert result.model == "fake-model"

    assert provider.received_query == "What is RAG?"
    assert provider.received_context is context
