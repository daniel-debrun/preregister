from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from preregister import AnalysisPlan, Comparison, Exclusion, Prediction, SpecError, Study
from preregister.canonical import sha256_hex

from .conftest import make_study


@pytest.mark.parametrize(
    ("expr", "kind", "conds", "op", "threshold", "direction"),
    [
        (
            "treatment - control > 0.05",
            "difference",
            ("treatment", "control"),
            ">",
            0.05,
            "increase",
        ),
        ("a > b", "difference", ("a", "b"), ">", 0.0, "increase"),
        ("a<b", "difference", ("a", "b"), "<", 0.0, "decrease"),
        ("new / old >= 1.1", "ratio", ("new", "old"), ">=", 1.1, "increase"),
        ("x > 0.5", "one_sample", ("x",), ">", 0.5, "increase"),
        ("x <= -1e-3", "one_sample", ("x",), "<=", -0.001, "decrease"),
    ],
)
def test_parse_comparisons(expr, kind, conds, op, threshold, direction) -> None:
    cmp = Comparison.parse(expr)
    assert (cmp.kind, cmp.conditions, cmp.op, cmp.threshold, cmp.direction) == (
        kind,
        conds,
        op,
        threshold,
        direction,
    )
    assert Comparison.parse(str(cmp)) == cmp


def test_parse_interval() -> None:
    cmp = Comparison.parse("x in [0.2, 0.4]")
    assert (cmp.kind, cmp.low, cmp.high, cmp.direction) == ("interval", 0.2, 0.4, "within")


@pytest.mark.parametrize(
    "expr", ["a - b", "a == b", "a > a", "a / b > 0", "a in [2, 1]", "a - b > c", "1 > a"]
)
def test_invalid_comparisons(expr: str) -> None:
    with pytest.raises(SpecError):
        Comparison.parse(expr)


def test_comparison_string_is_normalized_for_hashing() -> None:
    p1 = Prediction("P", "h", "m", "a-b>0.050", min_effect=1)
    p2 = Prediction("P", "h", "m", "a - b > 0.05", min_effect=1.0)
    assert p1.to_dict() == p2.to_dict()


def test_plan_validation() -> None:
    with pytest.raises(SpecError, match="Welch"):
        AnalysisPlan(test="welch", aggregation="iqm")
    with pytest.raises(SpecError):
        AnalysisPlan(alpha=0.7)
    with pytest.raises(SpecError):
        AnalysisPlan(n_seeds=3, seeds=(1, 2))
    with pytest.raises(SpecError):
        AnalysisPlan(n_seeds=2, seeds=(1, 1))
    assert AnalysisPlan(n_seeds=3).planned_seeds == (0, 1, 2)
    assert AnalysisPlan.from_dict({"seeds": [5, 6, 7]}).n_seeds == 3


def test_prediction_validation() -> None:
    with pytest.raises(SpecError, match="contradicts"):
        Prediction("P", "h", "m", "a > b", min_effect=1, direction="decrease")
    with pytest.raises(SpecError, match="min_effect"):
        Prediction("P", "h", "m", "a > b", min_effect=0)
    with pytest.raises(SpecError, match="interval"):
        Prediction(
            "P", "h", "m", "a in [0, 1]", min_effect=0.1, analysis=AnalysisPlan(test="permutation")
        )
    with pytest.raises(SpecError, match="two conditions"):
        Prediction(
            "P", "h", "m", "a > 1", min_effect=0.1, analysis=AnalysisPlan(test="mann_whitney")
        )


def test_exclusion_validation() -> None:
    assert Exclusion("status != finished", "crashed").parts == ("status", "!=", "finished")
    assert Exclusion("metric:loss is_nan", "diverged").parts == ("metric:loss", "is_nan", "")
    for rule in ("bogus == 1", "status !=", "metric is_nan 3"):
        with pytest.raises(SpecError):
            Exclusion(rule, "reason")
    with pytest.raises(SpecError, match="reason"):
        Exclusion("seed > 3", " ")


def test_study_validation(tmp_path: Path) -> None:
    base = make_study(tmp_path)
    bad_cond = Prediction("P2", "h", "score", "ghost - control > 0", min_effect=1)
    with pytest.raises(SpecError, match="unknown condition"):
        make_study(tmp_path, [base.predictions[0], bad_cond])
    bad_metric = Prediction("P2", "h", "nope", "treatment - control > 0", min_effect=1)
    with pytest.raises(SpecError, match="not declared"):
        make_study(tmp_path, [bad_metric])
    with pytest.raises(SpecError, match="duplicate"):
        make_study(tmp_path, [base.predictions[0], base.predictions[0]])
    mixed = [
        Prediction(
            "A", "h", "score", "treatment > control", 1, analysis=AnalysisPlan(correction="holm")
        ),
        Prediction(
            "B", "h", "score", "treatment > control", 1, analysis=AnalysisPlan(correction="bh")
        ),
    ]
    with pytest.raises(SpecError, match="mixes corrections"):
        make_study(tmp_path, mixed)


def test_yaml_matches_python_api(tmp_path: Path) -> None:
    spec = {
        "study": "test-study",
        "conditions": {
            "control": {"role": "control", "config": {"lr": 0.1}},
            "treatment": {"role": "treatment", "config": {"lr": 0.2}},
        },
        "metrics": {"score": {"version": "1"}},
        "predictions": [
            {
                "id": "P1",
                "hypothesis": "treatment beats control",
                "metric": "score",
                "comparison": "treatment - control > 0",
                "min_effect": 1,
                "analysis": {
                    "test": "welch",
                    "n_seeds": 10,
                    "n_boot": 2000,
                    "n_permutations": 2000,
                },
                "gates": [{"type": "MinSeeds", "n": 5}],
            }
        ],
    }
    path = tmp_path / "prereg.yaml"
    path.write_text(yaml.safe_dump(spec))
    from_yaml = Study.from_yaml(path)
    from_api = make_study(tmp_path)
    assert sha256_hex(from_yaml.frozen_payload("P1")) == sha256_hex(from_api.frozen_payload("P1"))


def test_frozen_payload_tracks_config_and_metric(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    h0 = sha256_hex(study.frozen_payload("P1"))
    assert h0 == sha256_hex(make_study(tmp_path).frozen_payload("P1"))
    study.conditions["control"] = study.conditions["control"].__class__(
        "control", role="control", config={"lr": 0.3}
    )
    assert sha256_hex(study.frozen_payload("P1")) != h0


def test_config_file_hash(tmp_path: Path) -> None:
    from preregister import Condition

    cfg = tmp_path / "c.yaml"
    cfg.write_text("lr: 0.1\n")
    cond = Condition("c", config_file="c.yaml")
    h = cond.config_hash(tmp_path)
    cfg.write_text("lr: 0.2\n")
    assert cond.config_hash(tmp_path) != h
    with pytest.raises(SpecError):
        Condition("c", config_file="missing.yaml").config_hash(tmp_path)


def test_unknown_yaml_keys_rejected(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="unknown keys"):
        Study.from_dict(
            {"study": "s", "conditions": {}, "metrics": {}, "predictons": []}, root=tmp_path
        )
