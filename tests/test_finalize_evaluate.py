import json
import shutil

import numpy as np
import pandas as pd
import pytest

from churnguard.config import PROCESSED_DIR, ROOT
from churnguard.data.split import load_split, split_xy
from churnguard.evaluation import evaluate as evaluate_module
from churnguard.evaluation.cost import Business
from churnguard.models import finalize as finalize_module
from churnguard.models.artifact import load_bundle, load_metadata

TUNING = ROOT / "reports" / "tuning_results.json"
needs_artifacts = pytest.mark.skipif(
    not ((PROCESSED_DIR / "val.parquet").exists() and TUNING.exists()),
    reason="run `make data` and `make train` first",
)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Run the finalize stage on a 500-row sample into a temp dir; record which splits it loaded."""
    root = tmp_path_factory.mktemp("final")
    loaded = []
    real_load = finalize_module.load_split
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(finalize_module, "load_split", lambda n: loaded.append(n) or real_load(n))
        meta = finalize_module.run_final(
            sample=500,
            reports_dir=root / "reports",
            model_path=root / "models" / "m.joblib",
            metadata_path=root / "models" / "metadata.json",
            tracking_uri=f"sqlite:///{root / 'mlruns' / 'mlflow.db'}",
        )
    return {"root": root, "meta": meta, "loaded": loaded}


@needs_artifacts
def test_finalize_never_loads_the_test_split(artifacts) -> None:
    assert sorted(artifacts["loaded"]) == ["train", "val"]


@needs_artifacts
def test_finalize_writes_a_usable_artifact_and_metadata(artifacts) -> None:
    root, meta = artifacts["root"], artifacts["meta"]
    bundle = load_bundle(root / "models" / "m.joblib")
    X, _ = split_xy(load_split("val"))
    proba = bundle["calibrated"].predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()
    assert bundle["base"].predict_proba(X).shape == (len(X), 2)

    on_disk = load_metadata(root / "models" / "metadata.json")
    assert on_disk["threshold"] == meta["threshold"]
    assert 0 < meta["threshold"] < 1
    assert meta["calibration"] in {"isotonic", "sigmoid"}
    assert meta["model_name"] in {"logistic_regression", "xgboost"}
    assert "test_metrics" not in on_disk
    for key in ("features", "model_input_features", "git_hash", "data_sha256", "library_versions"):
        assert on_disk[key]
    assert "assumption" in on_disk["business_assumptions"]["note"].lower()
    assert (root / "reports" / "figures" / "profit_vs_threshold.png").exists()
    assert (root / "reports" / "figures" / "calibration_validation.png").exists()


@needs_artifacts
def test_chosen_calibration_has_the_lowest_validation_brier(artifacts) -> None:
    cal = artifacts["meta"]["validation"]["calibration"]
    chosen = artifacts["meta"]["calibration"]
    assert cal[chosen]["brier"] == min(cal["isotonic"]["brier"], cal["sigmoid"]["brier"])


@needs_artifacts
def test_chosen_threshold_beats_target_everyone_and_nobody_on_validation(artifacts) -> None:
    s = artifacts["meta"]["validation"]["strategies"]
    assert s["chosen_threshold"]["net_value"] >= s["target_everyone"]["net_value"]
    assert s["chosen_threshold"]["net_value"] >= s["threshold_0.5"]["net_value"]
    assert s["chosen_threshold"]["net_value"] >= s["target_nobody"]["net_value"]


@needs_artifacts
def test_evaluate_dry_run_on_validation_split(artifacts) -> None:
    root = artifacts["root"]
    result = evaluate_module.run(
        split="val",
        reports_dir=root / "dry",
        model_path=root / "models" / "m.joblib",
        metadata_path=root / "models" / "metadata.json",
        log_to_mlflow=False,
    )
    assert result["split"] == "val" and not result["forced_rerun"]
    for lo, hi in result["bootstrap_95ci"].values():
        assert lo < hi
    assert set(result["slices"]) == {"contract", "tenure_bucket", "senior_citizen"}
    assert (root / "dry" / "figures" / "val_roc.png").exists()
    assert (root / "dry" / "val_evaluation.md").exists()
    assert "test_metrics" not in load_metadata(root / "models" / "metadata.json")


@needs_artifacts
def test_second_test_evaluation_is_refused_before_touching_data(artifacts, monkeypatch) -> None:
    root = artifacts["root"]
    reports = root / "guard"
    reports.mkdir()
    (reports / evaluate_module.SCORED_MARKER).write_text("{}")
    touched = []
    monkeypatch.setattr(evaluate_module, "load_split", lambda n: touched.append(n))
    with pytest.raises(evaluate_module.AlreadyEvaluatedError, match="--force"):
        evaluate_module.run(
            split="test",
            reports_dir=reports,
            model_path=root / "models" / "m.joblib",
            metadata_path=root / "models" / "metadata.json",
            log_to_mlflow=False,
        )
    assert touched == []


@needs_artifacts
def test_forced_rerun_is_recorded_and_updates_metadata(artifacts, monkeypatch) -> None:
    root = artifacts["root"]
    reports, meta_copy = root / "forced", root / "models" / "metadata_copy.json"
    reports.mkdir()
    (reports / evaluate_module.SCORED_MARKER).write_text("{}")
    shutil.copy(root / "models" / "metadata.json", meta_copy)
    monkeypatch.setattr(evaluate_module, "load_split", lambda n: load_split("val"))  # stand-in data
    result = evaluate_module.run(
        force=True,
        split="test",
        reports_dir=reports,
        model_path=root / "models" / "m.joblib",
        metadata_path=meta_copy,
        log_to_mlflow=False,
    )
    assert result["forced_rerun"] is True
    assert "test_metrics" in json.loads(meta_copy.read_text())
    assert json.loads((reports / "test_metrics.json").read_text())["forced_rerun"] is True


@needs_artifacts
def test_fresh_checkout_with_committed_results_may_score_once(artifacts, monkeypatch) -> None:
    """A clone receives test_metrics.json but no marker: it can reproduce once, then is guarded."""
    root = artifacts["root"]
    reports, meta_copy = root / "fresh", root / "models" / "metadata_fresh.json"
    reports.mkdir()
    (reports / "test_metrics.json").write_text("{}")  # committed results, but no marker
    shutil.copy(root / "models" / "metadata.json", meta_copy)
    monkeypatch.setattr(evaluate_module, "load_split", lambda n: load_split("val"))  # stand-in
    kwargs = {
        "split": "test",
        "reports_dir": reports,
        "model_path": root / "models" / "m.joblib",
        "metadata_path": meta_copy,
        "log_to_mlflow": False,
    }
    first = evaluate_module.run(**kwargs)
    assert first["forced_rerun"] is False
    assert (reports / evaluate_module.SCORED_MARKER).exists()
    with pytest.raises(evaluate_module.AlreadyEvaluatedError):
        evaluate_module.run(**kwargs)


def test_evaluate_model_pure_function_on_synthetic_data(random_customers) -> None:
    df = random_customers(n=400, seed=5)
    X, y = split_xy(df)
    rng = np.random.default_rng(0)
    proba = np.clip(0.2 + 0.5 * y.to_numpy() + rng.normal(0, 0.2, len(y)), 0.01, 0.99)
    res = evaluate_module.evaluate_model(
        proba, y.to_numpy(), X, 0.3, Business.from_config(), n_boot=100
    )
    assert res["metrics"]["roc_auc"] > 0.8
    assert res["metrics"]["tp"] + res["metrics"]["fn"] == int(y.sum())
    assert res["prd_targets"]["net_value_vs_target_everyone"]["target"] == "> 0"
    assert {s["group"] for s in res["slices"]["senior_citizen"]} <= {"Yes", "No"}
    assert isinstance(pd.DataFrame(res["slices"]["contract"]), pd.DataFrame)
    assert "PR-AUC" in evaluate_module.render_report(res, "val")
