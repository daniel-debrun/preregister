from __future__ import annotations

import math

import numpy as np
import pytest

from preregister.canonical import CanonicalizationError, canonical_json, sha256_hex


def test_key_order_and_whitespace_do_not_matter() -> None:
    a = {"b": [1, 2.5, {"z": None, "a": True}], "a": "x"}
    b = {"a": "x", "b": [1, 2.5, {"a": True, "z": None}]}
    assert canonical_json(a) == canonical_json(b) == '{"a":"x","b":[1,2.5,{"a":true,"z":null}]}'
    assert sha256_hex(a) == sha256_hex(b)


def test_known_digest_is_stable() -> None:
    obj = {"prediction": "P1", "alpha": 0.05}
    assert canonical_json(obj) == '{"alpha":0.05,"prediction":"P1"}'
    assert sha256_hex(obj) == "c90a4a002baa7cccd9f122efa95e12b6d6d64507d3afc33b74879f6ea9483994"


def test_tuples_numpy_and_negative_zero_normalize() -> None:
    assert canonical_json((1, 2)) == canonical_json([1, 2])
    assert canonical_json({"x": np.float64(0.25), "n": np.int64(3)}) == '{"n":3,"x":0.25}'
    assert canonical_json(-0.0) == canonical_json(0.0)


def test_unicode_is_preserved() -> None:
    assert canonical_json({"h": "effet > 0"}) == '{"h":"effet > 0"}'
    assert canonical_json("\u00e9t\u00e9") == '"\u00e9t\u00e9"'


@pytest.mark.parametrize("bad", [math.nan, math.inf, {1: "x"}, {"s": {1, 2}}])
def test_rejects_non_canonicalizable(bad: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json(bad)
