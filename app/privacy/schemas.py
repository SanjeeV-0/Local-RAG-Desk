from pydantic import BaseModel


class RedactionOutcome(BaseModel):
    """Always-present summary of whether sensitive content was withheld for
    this request -- mirrors `DecompositionOutcomeResponse`'s always-present
    pattern (see `app.retrieval.decomposition_outcome`).

    `redaction_applied`: true if any source chunk's content was redacted
    before being used for generation or returned to the caller.
    `affected_source_count`: how many sources had a redaction applied --
    never the sensitive values themselves.
    `answer_redacted`: true if the final generated answer itself contained a
    detectable PII span that had to be redacted after generation.
    `possibly_incomplete`: true whenever redaction may have withheld
    material relevant to the request -- set whenever `affected_source_count
    > 0` or `answer_redacted` is true. Callers must never treat a response
    with `possibly_incomplete=True` as a guaranteed-complete answer.
    """

    redaction_applied: bool = False
    affected_source_count: int = 0
    answer_redacted: bool = False
    possibly_incomplete: bool = False
