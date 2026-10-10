from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import ChunkPageMap, Document, DocumentPage, DocumentSection
from app.embeddings.testing import DeterministicEmbeddingProvider
from app.ingestion.chunk_service import ChunkService
from app.ingestion.size_guard import ChildChunk


class RecordingEmbeddingProvider(DeterministicEmbeddingProvider):
    """Wraps the deterministic test provider but records every text it was
    actually asked to embed, so a test can assert on the exact embedding
    INPUT `ChunkService.persist_chunks` constructs -- without needing the
    real sentence-transformers model."""

    def __init__(self, dimensions: int = 384) -> None:
        super().__init__(dimensions=dimensions)
        self.embedded_texts: list[str] = []

    def embed_batch(self, texts):
        texts = list(texts)
        self.embedded_texts.extend(texts)
        return super().embed_batch(texts)


@pytest.mark.asyncio
async def test_persist_chunks_maps_chunks_to_multiple_pages() -> None:
    async with async_session_factory() as session:
        async with async_session_factory() as session:
            document = Document(
                title=f"Embedding Test {uuid4()}",
                document_type="research_paper",
                content_hash=f"embedding-test-{uuid4()}",
                logical_document_id=uuid4(),
                version_number=1,
                is_current=True,
                document_metadata={},
            )

        session.add(document)
        await session.flush()

        page_one = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="Page one content.",
        )

        page_two = DocumentPage(
            document_id=document.id,
            page_number=2,
            content="Page two content.",
        )
        section = DocumentSection(
            document_id=document.id,
            title="Test Section",
            section_path="Test Section",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        session.add(section)
        await session.flush()

        session.add_all([page_one, page_two])
        await session.flush()

        page_ids = {
            1: page_one.id,
            2: page_two.id,
        }

        chunks = [
            ChildChunk(
                index=0,
                content="Chunk from page one.",
                page_numbers=[1],
                section_id=section.id,
                section_path="Test Section",
                section_level=1,
                source_units=(),
            ),
            ChildChunk(
                index=1,
                content="Chunk crossing pages.",
                page_numbers=[1, 2],
                section_id=section.id,
                section_path="Test Section",
                section_level=1,
                source_units=(),
            ),
        ]

        service = ChunkService(
            session,
            embedding_provider=DeterministicEmbeddingProvider(
                dimensions=384,
            ),
        )

        persisted_chunks = await service.persist_chunks(
            document_id=document.id,
            page_ids=page_ids,
            chunks=chunks,
        )

        await session.commit()

        assert len(persisted_chunks) == 2

        result = await session.execute(
            select(ChunkPageMap).where(ChunkPageMap.chunk_id == persisted_chunks[1].id)
        )

        mappings = result.scalars().all()

        assert len(mappings) == 2
        assert {mapping.document_page_id for mapping in mappings} == {
            page_one.id,
            page_two.id,
        }

        await session.delete(document)
        await session.commit()


@pytest.mark.asyncio
async def test_persist_chunks_embeds_heading_aware_text_but_stores_original_content() -> None:
    """End-to-end: `ChunkService.persist_chunks` must embed `section_path +
    content` (via `app.ingestion.embedding_text.build_embedding_text`), yet
    `DocumentChunk.content` persisted to the database must remain the
    ORIGINAL chunk text, byte for byte."""

    async with async_session_factory() as session:
        document = Document(
            title=f"Heading Aware Embedding Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"heading-aware-test-{uuid4()}",
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            document_metadata={},
        )
        session.add(document)
        await session.flush()

        section = DocumentSection(
            document_id=document.id,
            title="HNSW",
            section_path="Database > Indexing > HNSW",
            section_level=3,
            section_index=0,
            section_metadata={},
        )
        session.add(section)
        await session.flush()

        headed_content = "The graph is built with a configurable ef_construction parameter."
        root_content = "This paragraph has no owning section."

        chunks = [
            ChildChunk(
                index=0,
                content=headed_content,
                page_numbers=[],
                section_id=section.id,
                section_path="Database > Indexing > HNSW",
                section_level=3,
                source_units=(),
            ),
            ChildChunk(
                index=1,
                content=root_content,
                page_numbers=[],
                section_id=None,
                section_path="",
                section_level=0,
                source_units=(),
            ),
        ]

        provider = RecordingEmbeddingProvider(dimensions=384)
        service = ChunkService(session, embedding_provider=provider)

        persisted_chunks = await service.persist_chunks(
            document_id=document.id,
            page_ids={},
            chunks=chunks,
        )

        await session.commit()

        # The embedding model saw the heading-prefixed text...
        assert provider.embedded_texts == [
            "Database > Indexing > HNSW\n\n" + headed_content,
            root_content,
        ]

        # ...but the persisted, retrievable content is exactly the original
        # chunk text, with no heading prefix leaked into it.
        assert persisted_chunks[0].content == headed_content
        assert persisted_chunks[1].content == root_content

        await session.delete(document)
        await session.commit()
