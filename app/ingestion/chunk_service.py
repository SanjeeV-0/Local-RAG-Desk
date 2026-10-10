"""Final persistence step of the active ingestion pipeline: embeds and
stores `ChildChunk`s (from `app.ingestion.size_guard`) as `DocumentChunk`
rows, plus their `ChunkPageMap` provenance rows. The last ingestion step
before `DocumentService.mark_ready` promotes the version to READY/current.

Embedding input is heading-aware (see `app.ingestion.embedding_text.
build_embedding_text`) while `DocumentChunk.content` stores the chunk's
ORIGINAL text unchanged -- the embedding model sees the chunk's section
path as context, but nothing downstream (reranking, generation, the API)
ever sees anything other than the original content. Any future re-embedding
workflow MUST call the same `build_embedding_text` helper used here, never
a second, divergent formatting implementation, or old and new vectors would
not be directly comparable for reasons beyond just "the model changed".
"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import ChunkPageMap, DocumentChunk
from app.core.repositories.document import DocumentRepository
from app.embeddings.provider import EmbeddingProvider
from app.ingestion.embedding_text import build_embedding_text
from app.ingestion.size_guard import ChildChunk
from app.ingestion.structure import UnitType


class ChunkService:
    """Persists chunks, embeddings, and source-page mappings."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.repository = DocumentRepository(session)
        self.embedding_provider = embedding_provider

    async def persist_chunks(
        self,
        document_id: UUID,
        page_ids: dict[int, UUID],
        chunks: list[ChildChunk],
    ) -> list[DocumentChunk]:
        """Persist chunks with embeddings and source-page mappings."""

        if not chunks:
            return []

        embedding_texts = [
            build_embedding_text(chunk.section_path, chunk.content) for chunk in chunks
        ]

        embeddings = self.embedding_provider.embed_batch(embedding_texts)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding provider returned a different number of embeddings than chunks."
            )

        persisted_chunks: list[DocumentChunk] = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            # A table chunk's single source unit carries its
            # content_type/table_id/fragment metadata (attached by
            # fragment_table_units) -- surface it in chunk_metadata so a
            # table chunk is distinguishable from a prose chunk after
            # persistence, matching the `content_type: "table"` convention
            # already used by the (unwired) structure_chunker.CandidateGroup
            # metadata shape, kept consistent here on purpose.
            table_metadata: dict[str, object] = {}

            if len(chunk.source_units) == 1 and chunk.source_units[0].unit_type == UnitType.TABLE:
                table_metadata = dict(chunk.source_units[0].metadata)

            document_chunk = DocumentChunk(
                document_id=document_id,
                section_id=chunk.section_id,
                chunk_index=chunk.index,
                content=chunk.content,
                chunk_metadata={
                    "page_numbers": chunk.page_numbers,
                    "section_path": chunk.section_path,
                    **table_metadata,
                },
                embedding=embedding,
            )

            await self.repository.create_chunk(document_chunk)

            for page_number in chunk.page_numbers:
                page_id = page_ids[page_number]

                mapping = ChunkPageMap(
                    chunk_id=document_chunk.id,
                    document_page_id=page_id,
                )

                await self.repository.create_chunk_page_mapping(mapping)

            persisted_chunks.append(document_chunk)

        return persisted_chunks
