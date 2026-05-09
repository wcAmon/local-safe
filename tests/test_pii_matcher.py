from pipeline.pii.matcher import PIIMatcher
from pipeline.pii.tokens import PIIKind


def test_build_mapping_basic():
    m = PIIMatcher.build(
        entries=[("alice_92", PIIKind.USERNAME), ("新莊", PIIKind.LOCATION)],
        salt="t",
    )
    assert "alice_92" in m.raw_to_token
    assert "新莊" in m.raw_to_token
    assert m.raw_to_token["alice_92"].startswith("<<U-")
    assert m.raw_to_token["新莊"].startswith("<<LOC-")


def test_to_referenced_replaces_all_occurrences():
    m = PIIMatcher.build(
        entries=[("alice_92", PIIKind.USERNAME)], salt="t"
    )
    text = "alice_92 said something. then alice_92 again."
    out = m.to_referenced(text)
    assert "alice_92" not in out
    assert out.count(m.raw_to_token["alice_92"]) == 2


def test_redact_output_marks_leaks():
    m = PIIMatcher.build(
        entries=[("alice_92", PIIKind.USERNAME)], salt="t"
    )
    raw_output = "User alice_92 is friendly."
    redacted, leaked = m.redact_output(raw_output)
    token = m.raw_to_token["alice_92"]
    assert f"<<LEAKED:{token[2:-2]}>>" in redacted
    assert "alice_92" not in redacted
    assert leaked == [token]


def test_redact_output_no_leak_returns_empty_list():
    m = PIIMatcher.build(
        entries=[("alice_92", PIIKind.USERNAME)], salt="t"
    )
    raw_output = "User user_001 is friendly."
    redacted, leaked = m.redact_output(raw_output)
    assert leaked == []
    assert redacted == raw_output


def test_redact_substring_partial_leak():
    """If model writes 'alice' (substring of 'alice_92'), flag as partial."""
    m = PIIMatcher.build(
        entries=[("alice_92", PIIKind.USERNAME)], salt="t"
    )
    raw_output = "User alice is friendly."
    redacted, leaked = m.redact_output(raw_output, partial=True)
    token = m.raw_to_token["alice_92"]
    assert f"<<PARTIAL_LEAK:{token[2:-2]}>>" in redacted
    assert leaked == [token]


def test_redact_full_overrides_partial():
    """When raw_output contains the full username, full leak wins over partial."""
    m = PIIMatcher.build(
        entries=[("alice_92", PIIKind.USERNAME)], salt="t"
    )
    redacted, leaked = m.redact_output("hi alice_92", partial=True)
    token = m.raw_to_token["alice_92"]
    assert f"<<LEAKED:{token[2:-2]}>>" in redacted
    assert "<<PARTIAL_LEAK:" not in redacted


def test_extract_known_refs_finds_present():
    m = PIIMatcher.build(
        entries=[("alice_92", PIIKind.USERNAME), ("新莊", PIIKind.LOCATION)],
        salt="t",
    )
    refs = m.extract_known_refs("alice_92 went to 新莊 today")
    assert refs == {m.raw_to_token["alice_92"], m.raw_to_token["新莊"]}


def test_extract_known_refs_returns_empty_for_no_match():
    m = PIIMatcher.build(entries=[("alice_92", PIIKind.USERNAME)], salt="t")
    refs = m.extract_known_refs("no PII here")
    assert refs == set()


def test_extract_known_refs_ignores_token_strings_themselves():
    """If text already contains <<U-...>> tokens (from referenced sample),
    we should not double-count them as new exposures."""
    m = PIIMatcher.build(entries=[("alice_92", PIIKind.USERNAME)], salt="t")
    tok = m.raw_to_token["alice_92"]
    refs = m.extract_known_refs(f"already-tokenized: {tok}")
    # The token string itself is not the raw — should not match.
    # alice_92 is not in the text either. So empty.
    assert refs == set()
