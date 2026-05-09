from pipeline.pii.tokens import mint_token, PIIKind


def test_mint_token_is_deterministic():
    t1 = mint_token("alice_92", PIIKind.USERNAME, salt="proj-salt")
    t2 = mint_token("alice_92", PIIKind.USERNAME, salt="proj-salt")
    assert t1 == t2


def test_mint_token_differs_with_salt():
    t1 = mint_token("alice_92", PIIKind.USERNAME, salt="salt-a")
    t2 = mint_token("alice_92", PIIKind.USERNAME, salt="salt-b")
    assert t1 != t2


def test_mint_token_different_kinds_distinct():
    t1 = mint_token("alice_92", PIIKind.USERNAME, salt="s")
    t2 = mint_token("alice_92", PIIKind.LOCATION, salt="s")
    assert t1 != t2


def test_token_format():
    t = mint_token("alice_92", PIIKind.USERNAME, salt="s")
    # e.g. <<U-7f3a2c>>
    assert t.startswith("<<U-")
    assert t.endswith(">>")
    assert len(t) == len("<<U-XXXXXX>>")  # 6-hex truncated


def test_all_prefixes():
    samples = [
        (PIIKind.USERNAME, "U-"),
        (PIIKind.LOCATION, "LOC-"),
        (PIIKind.ORGANIZATION, "ORG-"),
        (PIIKind.WRITING_STYLE, "STYLE-"),
        (PIIKind.OCCUPATION, "OCC-"),
        (PIIKind.TIME_PATTERN, "TIME-"),
    ]
    for kind, prefix in samples:
        t = mint_token("x", kind, salt="s")
        assert t.startswith(f"<<{prefix}")
