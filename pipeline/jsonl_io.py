"""JSONL artifact I/O with pydantic validation and idempotent append."""

from pathlib import Path
from typing import Iterator, Iterable, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path, model_cls: type[T]) -> Iterator[T]:
    """Yield validated pydantic objects from a JSONL file. Blank lines skipped."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            yield model_cls.model_validate_json(line)


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> None:
    """Overwrite path with serialized JSONL of items. Atomic via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json())
            fh.write("\n")
    tmp.replace(path)


def append_jsonl_idempotent(
    path: Path, items: Iterable[BaseModel], *, key: str
) -> int:
    """Append items whose `key` field is not already present in the file.

    Returns the number of items actually appended. The dedupe is by file scan;
    for very large artifacts this is O(N) per append batch, which is fine for
    Phase 1 sizes (<= 100k rows).
    """
    existing_keys: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                # Parse only the key field, not the full record; cheap path.
                import json
                obj = json.loads(line)
                existing_keys.add(obj[key])

    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for item in items:
            k = getattr(item, key)
            if k in existing_keys:
                continue
            fh.write(item.model_dump_json())
            fh.write("\n")
            existing_keys.add(k)
            n += 1
    return n
