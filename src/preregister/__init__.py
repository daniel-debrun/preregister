"""preregister: pre-registration for machine learning experiments."""

from preregister.gates import GateContext, GateResult, custom_gate
from preregister.ledger import Ledger, LedgerEntry, VerifyReport
from preregister.project import LinkSummary, Project, RegistrationError
from preregister.reconcile import PredictionResult, StudyAnalysis, Verdict
from preregister.registry import (
    AnalysisPlan,
    Comparison,
    Condition,
    Exclusion,
    ExploratorySpec,
    GateSpec,
    Metric,
    Prediction,
    SpecError,
    Study,
)
from preregister.runs import RunRecord, load_runs_csv, load_runs_json

__version__ = "0.1.0"

__all__ = [
    "AnalysisPlan",
    "Comparison",
    "Condition",
    "Exclusion",
    "ExploratorySpec",
    "GateContext",
    "GateResult",
    "GateSpec",
    "Ledger",
    "LedgerEntry",
    "LinkSummary",
    "Metric",
    "Prediction",
    "PredictionResult",
    "Project",
    "RegistrationError",
    "RunRecord",
    "SpecError",
    "Study",
    "StudyAnalysis",
    "Verdict",
    "VerifyReport",
    "__version__",
    "custom_gate",
    "load_runs_csv",
    "load_runs_json",
]
