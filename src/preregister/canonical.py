"""Canonical JSON serialization and hashing.

Every frozen object in preregister is hashed as SHA-256 over a canonical JSON encoding:
sorted keys, no insignificant whitespace, UTF-8, no NaN/Infinity, tuples as lists,
numpy scalars as Python scalars, and ``-0.0`` normalized to ``0.0``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CanonicalizationError(ValueError):
    pass


def _normalize(obj: Any, path: str = "$") -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise CanonicalizationError(f"non-finite float at {path}: {obj!r}")
        return 0.0 if obj == 0.0 else obj
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"non-string key at {path}: {key!r}")
            out[key] = _normalize(value, f"{path}.{key}")
        return out
    if isinstance(obj, (list, tuple)):
        return [_normalize(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    item = getattr(obj, "item", None)
    if callable(item) and type(obj).__module__ == "numpy":
        return _normalize(item(), path)
    raise CanonicalizationError(f"unsupported type at {path}: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    return json.dumps(
        _normalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()
