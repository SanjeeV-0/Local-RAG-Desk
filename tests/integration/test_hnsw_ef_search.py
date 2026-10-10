"""Verifies that `DocumentRepository.search_similar_chunks` applies a
configured `hnsw.ef_search` via a per-transaction `SET LOCAL`, scoped so it
can never leak across pooled-connection reuse -- see the setting's docstring
in `app.config.settings.Settings.hnsw_ef_search` and the scoping rationale
in `DocumentRepository.search_similar_chunks`.

These tests perform NO writes -- they only run `search_similar_chunks`
(a read-only query) against whatever rows already exist (even zero rows is
a valid, successful result) and inspect the session's own in-transaction
GUC state via `SHOW`. There is nothing to clean up and nothing that could
touch pre-existing data, so the usual test-ownership/cleanup concerns for
writing tests do not apply here.
"""

from uuid import uuid4

import pytest
from sqlalchemy import text

from app.config.settings import get_settings
from app.core.database import async_session_factory
from app.core.repositories.document import DocumentRepository


def test_database_target_is_the_configured_development_database() -> None:
    """Sanity check before any DB-touching test in this file: confirms we
    are pointed at the single configured database, not something unknown
    -- without printing credentials."""

    database_url = get_settings().database_url

    assert "research_agent" in database_url
    assert "localhost" in database_url


@pytest.mark.asyncio
async def test_search_similar_chunks_applies_configured_ef_search() -> None:
    async with async_session_factory() as session:
        repository = DocumentRepository(session, hnsw_ef_search=77)

        await repository.search_similar_chunks(
            query_embedding=[0.1] * 384,
            limit=1,
        )

        # Still inside the SAME transaction `search_similar_chunks` ran
        # its SET LOCAL in -- this is the direct proof the GUC was actually
        # applied to the connection executing the query, not just
        # constructed and ignored.
        result = await session.execute(text("SHOW hnsw.ef_search"))
        assert result.scalar() == "77"

        await session.rollback()


@pytest.mark.asyncio
async def test_search_similar_chunks_does_not_touch_ef_search_when_unconfigured() -> None:
    async with async_session_factory() as session:
        repository = DocumentRepository(session, hnsw_ef_search=None)

        await repository.search_similar_chunks(
            query_embedding=[0.2] * 384,
            limit=1,
        )

        # No SET LOCAL was issued -- whatever the server/session default is,
        # it must not be our test sentinel value from the other test.
        result = await session.execute(text("SHOW hnsw.ef_search"))
        assert result.scalar() != "77"

        await session.rollback()


@pytest.mark.asyncio
async def test_ef_search_does_not_leak_across_transactions_on_reused_connection() -> None:
    """The core pooled-connection safety guarantee: configuring ef_search
    for one request's transaction must not affect a LATER transaction that
    happens to reuse the same underlying connection -- simulated here by
    reusing the same session (and therefore likely the same physical
    connection) across two sequential, separately-committed transactions."""

    async with async_session_factory() as session:
        configured_repository = DocumentRepository(session, hnsw_ef_search=123)

        await configured_repository.search_similar_chunks(
            query_embedding=[0.3] * 384,
            limit=1,
        )

        mid_transaction = await session.execute(text("SHOW hnsw.ef_search"))
        assert mid_transaction.scalar() == "123"

        # Ends the transaction SET LOCAL was scoped to.
        await session.commit()

        unconfigured_repository = DocumentRepository(session, hnsw_ef_search=None)

        await unconfigured_repository.search_similar_chunks(
            query_embedding=[0.4] * 384,
            limit=1,
        )

        after_commit = await session.execute(text("SHOW hnsw.ef_search"))
        assert after_commit.scalar() != "123"

        await session.rollback()


@pytest.mark.asyncio
async def test_search_similar_chunks_works_with_filters_and_configured_ef_search() -> None:
    """Confirms the SET LOCAL addition composes correctly with the existing
    document_id/section_id/max_distance filters and limit -- no behavioral
    change to the query itself, just an extra statement ahead of it."""

    async with async_session_factory() as session:
        repository = DocumentRepository(session, hnsw_ef_search=10)

        results = await repository.search_similar_chunks(
            query_embedding=[0.5] * 384,
            limit=5,
            max_distance=0.99,
            document_id=uuid4(),
            section_id=uuid4(),
        )

        assert results == []

        await session.rollback()


@pytest.mark.asyncio
async def test_search_similar_chunks_applies_configured_iterative_scan() -> None:
    async with async_session_factory() as session:
        repository = DocumentRepository(session, hnsw_iterative_scan="relaxed_order")

        await repository.search_similar_chunks(query_embedding=[0.6] * 384, limit=1)

        result = await session.execute(text("SHOW hnsw.iterative_scan"))
        assert result.scalar() == "relaxed_order"

        await session.rollback()


@pytest.mark.asyncio
async def test_search_similar_chunks_does_not_touch_iterative_scan_when_unconfigured() -> None:
    async with async_session_factory() as session:
        repository = DocumentRepository(session, hnsw_iterative_scan=None)

        await repository.search_similar_chunks(query_embedding=[0.65] * 384, limit=1)

        result = await session.execute(text("SHOW hnsw.iterative_scan"))
        # pgvector's own built-in default.
        assert result.scalar() == "off"

        await session.rollback()


@pytest.mark.asyncio
async def test_search_similar_chunks_applies_configured_max_scan_tuples() -> None:
    async with async_session_factory() as session:
        repository = DocumentRepository(session, hnsw_max_scan_tuples=12345)

        await repository.search_similar_chunks(query_embedding=[0.7] * 384, limit=1)

        result = await session.execute(text("SHOW hnsw.max_scan_tuples"))
        assert result.scalar() == "12345"

        await session.rollback()


@pytest.mark.asyncio
async def test_search_similar_chunks_does_not_touch_max_scan_tuples_when_unconfigured() -> None:
    async with async_session_factory() as session:
        repository = DocumentRepository(session, hnsw_max_scan_tuples=None)

        await repository.search_similar_chunks(query_embedding=[0.75] * 384, limit=1)

        result = await session.execute(text("SHOW hnsw.max_scan_tuples"))
        # pgvector's own built-in default.
        assert result.scalar() == "20000"

        await session.rollback()


@pytest.mark.asyncio
async def test_search_similar_chunks_applies_configured_scan_mem_multiplier() -> None:
    async with async_session_factory() as session:
        repository = DocumentRepository(session, hnsw_scan_mem_multiplier=4)

        await repository.search_similar_chunks(query_embedding=[0.8] * 384, limit=1)

        result = await session.execute(text("SHOW hnsw.scan_mem_multiplier"))
        assert result.scalar() == "4"

        await session.rollback()


@pytest.mark.asyncio
async def test_search_similar_chunks_does_not_touch_scan_mem_multiplier_when_unconfigured() -> None:
    async with async_session_factory() as session:
        repository = DocumentRepository(session, hnsw_scan_mem_multiplier=None)

        await repository.search_similar_chunks(query_embedding=[0.85] * 384, limit=1)

        result = await session.execute(text("SHOW hnsw.scan_mem_multiplier"))
        # pgvector's own built-in default.
        assert result.scalar() == "1"

        await session.rollback()


@pytest.mark.asyncio
async def test_search_similar_chunks_applies_all_four_hnsw_settings_together() -> None:
    """All four GUCs can be configured simultaneously without interfering
    with each other or with the query itself."""

    async with async_session_factory() as session:
        repository = DocumentRepository(
            session,
            hnsw_ef_search=200,
            hnsw_iterative_scan="strict_order",
            hnsw_max_scan_tuples=5000,
            hnsw_scan_mem_multiplier=2,
        )

        results = await repository.search_similar_chunks(query_embedding=[0.9] * 384, limit=1)
        assert results == []

        assert (await session.execute(text("SHOW hnsw.ef_search"))).scalar() == "200"
        assert (await session.execute(text("SHOW hnsw.iterative_scan"))).scalar() == "strict_order"
        assert (await session.execute(text("SHOW hnsw.max_scan_tuples"))).scalar() == "5000"
        assert (await session.execute(text("SHOW hnsw.scan_mem_multiplier"))).scalar() == "2"

        await session.rollback()


@pytest.mark.asyncio
async def test_none_of_the_four_hnsw_settings_leak_across_a_commit() -> None:
    async with async_session_factory() as session:
        configured = DocumentRepository(
            session,
            hnsw_ef_search=321,
            hnsw_iterative_scan="strict_order",
            hnsw_max_scan_tuples=999,
            hnsw_scan_mem_multiplier=9,
        )
        await configured.search_similar_chunks(query_embedding=[0.95] * 384, limit=1)
        await session.commit()

        unconfigured = DocumentRepository(session)
        await unconfigured.search_similar_chunks(query_embedding=[0.99] * 384, limit=1)

        assert (await session.execute(text("SHOW hnsw.ef_search"))).scalar() != "321"
        assert (await session.execute(text("SHOW hnsw.iterative_scan"))).scalar() == "off"
        assert (await session.execute(text("SHOW hnsw.max_scan_tuples"))).scalar() == "20000"
        assert (await session.execute(text("SHOW hnsw.scan_mem_multiplier"))).scalar() == "1"

        await session.rollback()
