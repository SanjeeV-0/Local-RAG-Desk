import pytest

from app.api.research import _execute_research
from app.generation.context import GenerationContext
from app.generation.generation_service import GenerationResult
from app.retrieval.decomposition_outcome import DecompositionOutcome
from app.retrieval.models import RetrievedChunk


class FakeRetrievalService:
    def __init__(self) -> None:
        self.last_decomposition_outcome: DecompositionOutcome | None = None

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[RetrievedChunk]:
        # Always set, exactly like the real `RetrievalService._search`.
        self.last_decomposition_outcome = DecompositionOutcome(
            original_query=query,
            sub_queries=[query],
            status="not_needed",
        )

        return [
            RetrievedChunk(
                document_id=__import__("uuid").uuid4(),
                chunk_id=__import__("uuid").uuid4(),
                section_id=__import__("uuid").uuid4(),
                section_path="Results",
                page_numbers=[1],
                content="Research evidence.",
                distance=0.1,
            )
        ]


class FakeGenerationService:
    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        return GenerationResult(
            text="Generated research answer.",
            model="fake-model",
            input_tokens=50,
            output_tokens=15,
            total_tokens=65,
        )


class FakeObservation:
    def __init__(self):
        self.output = None
        self.level = None
        self.status_message = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(
        self,
        *,
        output=None,
        level=None,
        status_message=None,
    ):
        self.output = output
        self.level = level
        self.status_message = status_message


class FakeLangfuse:
    def __init__(self):
        self.observation_type = None
        self.name = None
        self.input = None
        self.observation = None

    def start_as_current_observation(
        self,
        *,
        as_type,
        name,
        input,
    ):
        self.observation_type = as_type
        self.name = name
        self.input = input

        self.observation = FakeObservation()

        return self.observation


@pytest.mark.asyncio
async def test_research_execution_returns_answer_and_sources():
    response = await _execute_research(
        query="What is RAG?",
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
    )

    assert response["answer"] == "Generated research answer."
    assert response["model"] == "fake-model"
    assert len(response["sources"]) == 1
    assert response["sources"][0]["section_path"] == "Results"
