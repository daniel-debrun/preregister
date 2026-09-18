"""Adapters that pull runs from experiment trackers. Tracker packages are imported lazily."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from preregister.canonical import sha256_hex


def lookup(data: Mapping[str, Any], dotted: str) -> Any:
    """Resolve ``a.b.c`` in nested mappings; a literal key containing dots wins."""
    if dotted in data:
        return data[dotted]
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def subset_config_hash(config: Mapping[str, Any], keys: Sequence[str] | None) -> str | None:
    """Hash of the selected config keys, comparable with ``Condition(config=...)`` hashes."""
    if not keys:
        return None
    return sha256_hex({k: config.get(k) for k in keys})
