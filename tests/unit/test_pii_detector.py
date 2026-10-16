from app.privacy.pii_detector import detect, redact

# All values below are synthetic markers for testing only -- not real PII.
SYNTHETIC_EMAIL = "jdoe@example.com"
SYNTHETIC_PHONE = "555-123-4567"
SYNTHETIC_SSN = "123-45-6789"
# Luhn-valid synthetic test card number (standard "test" Visa number).
SYNTHETIC_CARD = "4111 1111 1111 1111"
SYNTHETIC_CARD_NO_SPACES = "4111111111111111"


def test_detects_email():
    matches = detect(f"Contact: {SYNTHETIC_EMAIL} for details.")
    assert len(matches) == 1
    assert matches[0].category == "email"


def test_detects_phone():
    matches = detect(f"Call us at {SYNTHETIC_PHONE} anytime.")
    assert len(matches) == 1
    assert matches[0].category == "phone"


def test_detects_dashed_ssn():
    matches = detect(f"SSN on file: {SYNTHETIC_SSN}.")
    assert len(matches) == 1
    assert matches[0].category == "ssn"


def test_bare_nine_digit_run_is_not_treated_as_ssn():
    # Deliberate limitation: undashed 9-digit runs are not matched as SSNs
    # (too false-positive-prone against arbitrary numeric IDs).
    matches = detect("Reference number 123456789 on the invoice.")
    assert matches == []


def test_detects_luhn_valid_card_with_spaces():
    matches = detect(f"Card: {SYNTHETIC_CARD}")
    assert len(matches) == 1
    assert matches[0].category == "credit_card"


def test_detects_luhn_valid_card_without_spaces():
    matches = detect(f"Card: {SYNTHETIC_CARD_NO_SPACES}")
    assert len(matches) == 1
    assert matches[0].category == "credit_card"


def test_luhn_invalid_long_digit_run_is_not_flagged_as_card():
    # 17-digit ISBN-like number that fails the Luhn checksum -- documented
    # false-positive avoidance, not a card.
    matches = detect("ISBN-ish code: 12345678901234567")
    assert matches == []


def test_international_phone_format_is_a_known_false_negative():
    # Documented limitation: this detector's phone pattern does not cover
    # all international formats. This test captures current (incomplete)
    # behavior, not a claim of correctness.
    matches = detect("Call +44 20 7946 0958 for support.")
    assert matches == []


def test_redact_preserves_surrounding_safe_text():
    text = f"Please email {SYNTHETIC_EMAIL} before Friday's meeting."
    result = redact(text)

    assert result.redacted is True
    assert "[REDACTED:EMAIL]" in result.text
    assert SYNTHETIC_EMAIL not in result.text
    assert "Please email" in result.text
    assert "before Friday's meeting." in result.text


def test_redact_mixed_content_chunk_preserves_multiple_safe_segments():
    text = (
        f"The on-call contact is {SYNTHETIC_EMAIL}. "
        "The deployment runbook is in the wiki. "
        f"Escalate by phone at {SYNTHETIC_PHONE} if urgent."
    )
    result = redact(text)

    assert result.categories_found == {"email", "phone"}
    assert SYNTHETIC_EMAIL not in result.text
    assert SYNTHETIC_PHONE not in result.text
    assert "The deployment runbook is in the wiki." in result.text


def test_redact_no_match_returns_original_text_unchanged():
    text = "This document contains no sensitive information at all."
    result = redact(text)

    assert result.redacted is False
    assert result.text == text
    assert result.matches == ()
    assert result.categories_found == frozenset()


def test_overlapping_matches_prefer_longer_and_higher_priority_span():
    # A card-shaped number embedded right after an SSN-shaped substring --
    # ensure resolution doesn't double-count/corrupt either.
    text = f"SSN {SYNTHETIC_SSN} and card {SYNTHETIC_CARD_NO_SPACES} on file."
    result = redact(text)

    assert result.categories_found == {"ssn", "credit_card"}
    assert SYNTHETIC_SSN not in result.text
    assert SYNTHETIC_CARD_NO_SPACES not in result.text


def test_multiple_same_category_matches_all_redacted():
    other_email = "other.user@example.org"
    text = f"From {SYNTHETIC_EMAIL} to {other_email}."
    result = redact(text)

    assert len(result.matches) == 2
    assert SYNTHETIC_EMAIL not in result.text
    assert other_email not in result.text
