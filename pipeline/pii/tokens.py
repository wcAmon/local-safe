"""Deterministic, salted PII token minting."""

import hashlib
from enum import Enum


class PIIKind(str, Enum):
    USERNAME = "username"
    LOCATION = "location"
    ORGANIZATION = "organization"
    WRITING_STYLE = "writing_style"
    OCCUPATION = "occupation"
    TIME_PATTERN = "time_pattern"


_PREFIX = {
    PIIKind.USERNAME: "U",
    PIIKind.LOCATION: "LOC",
    PIIKind.ORGANIZATION: "ORG",
    PIIKind.WRITING_STYLE: "STYLE",
    PIIKind.OCCUPATION: "OCC",
    PIIKind.TIME_PATTERN: "TIME",
}


def mint_token(raw: str, kind: PIIKind, *, salt: str) -> str:
    """Return the opaque token for a raw PII string under a given kind.

    Format: ``<<PREFIX-XXXXXX>>`` where ``XXXXXX`` is the first 6 hex chars
    of ``sha256(salt|kind|raw)``.

    Deterministic given (raw, kind, salt). The salt prevents cross-project
    correlation; the kind prefix prevents collisions across PII categories.
    """
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(b"|")
    h.update(kind.value.encode("utf-8"))
    h.update(b"|")
    h.update(raw.encode("utf-8"))
    digest = h.hexdigest()[:6]
    return f"<<{_PREFIX[kind]}-{digest}>>"
