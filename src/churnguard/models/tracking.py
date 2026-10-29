"""MLflow helpers: local file tracking, provenance tags, one call to log a CV result."""

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pandas as pd

from churnguard.config import RAW_PATH, ROOT

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def data_checksum() -> str:
    """SHA256 recorded by scripts/download_data.py ('unknown' if the file is missing)."""
    path = RAW_PATH.with_name(RAW_PATH.name + ".sha256")
    return path.read_text().split()[0] if path.exists() else "unknown"


def git_hash() -> str:
    """Short HEAD hash, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def setup(tracking_uri: str, experiment: str) -> None:
    """Point MLflow at a local SQLite store with artifacts next to it.

    MLflow 3 rejects the plain file store, so only `sqlite:///<path>` is supported;
    a relative path resolves against the repo root.
    """
    if not tracking_uri.startswith("sqlite:///"):
        raise ValueError(
            f"tracking_uri must look like sqlite:///mlruns/mlflow.db, got {tracking_uri}"
        )
    db = Path(tracking_uri.removeprefix("sqlite:///"))
    db = db if db.is_absolute() else ROOT / db
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(db.parent / "artifacts").as_uri())
    mlflow.set_experiment(experiment)


@contextmanager
def run(name: str, params: dict, nested: bool = False):  # noqa: ANN201
    """Start an MLflow run tagged with data checksum and git hash, logging `params`."""
    with mlflow.start_run(run_name=name, nested=nested) as active:
        mlflow.set_tags({"data_sha256": data_checksum(), "git_hash": git_hash()})
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        yield active


def log_cv(summary: dict[str, float], fold_scores: pd.DataFrame) -> None:
    """Log CV mean/std metrics and the per-fold table to the active run."""
    mlflow.log_metrics(summary)
    mlflow.log_table(fold_scores, "fold_scores.json")
