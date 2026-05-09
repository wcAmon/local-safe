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
