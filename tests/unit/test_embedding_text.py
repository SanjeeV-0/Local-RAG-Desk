from app.ingestion.embedding_text import build_embedding_text


def test_single_heading_is_prepended() -> None:
    result = build_embedding_text("Results", "The model achieved 92% accuracy.")

    assert result == "Results\n\nThe model achieved 92% accuracy."


def test_nested_heading_path_is_prepended_verbatim() -> None:
    result = build_embedding_text(
        "Database > Indexing > HNSW",
        "HNSW builds a multi-layer graph for approximate nearest neighbor search.",
    )

    assert result == (
        "Database > Indexing > HNSW\n\n"
        "HNSW builds a multi-layer graph for approximate nearest neighbor search."
    )


def test_missing_heading_returns_content_unchanged_for_none() -> None:
    content = "Orphaned paragraph with no owning section."

    assert build_embedding_text(None, content) == content


def test_missing_heading_returns_content_unchanged_for_empty_string() -> None:
    """Root-level chunks store `section_path=""` (see
    `app.ingestion.structure_extractor`), not `None` -- both must behave
    identically: no fabricated heading, content passed through as-is."""

    content = "Root-level content with no preceding heading."

    assert build_embedding_text("", content) == content


def test_original_content_is_never_mutated_or_truncated_by_the_helper() -> None:
    content = "Multi-line content.\n\nWith a blank-line-separated second paragraph."

    result = build_embedding_text("Section", content)

    assert content in result
    assert result.endswith(content)


def test_deterministic_for_identical_inputs() -> None:
    first = build_embedding_text("A > B", "Some content.")
    second = build_embedding_text("A > B", "Some content.")

    assert first == second


def test_different_section_paths_produce_different_embedding_text() -> None:
    """Same content under two different headings must not collapse to the
    same embedding input -- the whole point of heading-aware embeddings."""

    content = "Install the package and run the server."

    installation_variant = build_embedding_text("Installation", content)
    deployment_variant = build_embedding_text("Deployment > Production", content)

    assert installation_variant != deployment_variant
    assert content in installation_variant
    assert content in deployment_variant


def test_repeated_heading_title_at_different_paths_is_not_conflated() -> None:
    content = "See the overview above."

    first = build_embedding_text("Intro > Overview", content)
    second = build_embedding_text("Setup > Overview", content)

    assert first != second
