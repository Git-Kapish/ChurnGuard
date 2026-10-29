import numpy as np
import pandas as pd
import pytest

from churnguard.config import PROCESSED_DIR, ROOT
from churnguard.data.drift import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    drift_report,
    psi,
    psi_categorical,
    psi_numeric,
)
from churnguard.data.split import load_split

needs_data = pytest.mark.skipif(
    not (PROCESSED_DIR / "train.parquet").exists(), reason="run `make data` first"
)


def test_psi_matches_hand_computation() -> None:
    # (0.8-0.5)*ln(0.8/0.5) + (0.2-0.5)*ln(0.2/0.5) = 0.3*ln(1.6) + 0.3*ln(2.5)
    assert psi([0.5, 0.5], [0.8, 0.2]) == pytest.approx(0.3 * np.log(1.6) + 0.3 * np.log(2.5))


def test_psi_is_zero_for_identical_and_scale_invariant() -> None:
    assert psi([30, 70], [30, 70]) == pytest.approx(0.0)
    assert psi([30, 70], [3, 7]) == pytest.approx(0.0)  # counts or shares: same answer
    assert psi([1, 2, 3], [3, 2, 1]) == pytest.approx(psi([3, 2, 1], [1, 2, 3]))  # symmetric


def test_psi_stays_finite_with_empty_bins() -> None:
    value = psi([50, 50, 0], [0, 50, 50])
    assert np.isfinite(value) and value > 0.2


def test_numeric_psi_detects_a_shift_but_not_resampling() -> None:
    rng = np.random.default_rng(0)
    train = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(train, pd.Series(rng.normal(0, 1, 2000))) < 0.05
    assert psi_numeric(train, pd.Series(rng.normal(1.5, 1, 2000))) > 0.2


def test_numeric_psi_handles_out_of_range_and_constant_values() -> None:
    train = pd.Series(np.arange(1000, dtype=float))
    far_above = pd.Series(np.full(500, 1e6))  # all land in the last bin, nothing crashes
    assert psi_numeric(train, far_above) > 0.2
    assert psi_numeric(pd.Series(np.full(100, 5.0)), pd.Series(np.full(50, 7.0))) == 0.0
    assert psi_numeric(train, pd.Series([np.nan, 5.0, 6.0] * 100)) >= 0  # NaNs ignored


def test_categorical_psi_detects_mix_shift_and_unseen_categories() -> None:
    train = pd.Series(["a"] * 600 + ["b"] * 400)
    assert psi_categorical(train, pd.Series(["a"] * 60 + ["b"] * 40)) == pytest.approx(0.0)
    assert psi_categorical(train, pd.Series(["a"] * 20 + ["b"] * 80)) > 0.2
    assert psi_categorical(train, pd.Series(["a"] * 50 + ["b"] * 30 + ["new"] * 20)) > 0.2


def test_drift_report_flags_exactly_the_shifted_features(random_customers) -> None:
    train = random_customers(n=2000, seed=1)
    stable = random_customers(n=800, seed=2)
    assert not drift_report(train, stable)["flag"].any()

    shifted = stable.assign(Contract="Month-to-month", MonthlyCharges=stable["MonthlyCharges"] + 60)
    report = drift_report(train, shifted)
    assert set(report.loc[report["flag"], "feature"]) >= {"Contract", "MonthlyCharges"}
    assert report.iloc[0]["psi"] == report["psi"].max()  # sorted, largest first
    assert set(report["feature"]) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_drift_report_accepts_the_raw_dataset_vocabulary(random_customers) -> None:
    train = random_customers(n=500, seed=1)
    raw = train.assign(MultipleLines="No phone service", TechSupport="No internet service")
    cleaned = train.assign(MultipleLines="No", TechSupport="No")
    pd.testing.assert_frame_equal(drift_report(train, raw), drift_report(train, cleaned))


@needs_data
def test_validation_split_has_no_drift_against_training() -> None:
    report = drift_report(load_split("train"), load_split("val"))
    assert not report["flag"].any() and report["psi"].max() < 0.1  # same stratified population


@needs_data
def test_cli_exit_codes_and_report_file(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import drift_check

    val = load_split("val")
    assert drift_check.main([str(PROCESSED_DIR / "val.parquet")]) == 0

    drifted = val.assign(Contract="Month-to-month", MonthlyCharges=val["MonthlyCharges"] * 3)
    csv, out = tmp_path / "new.csv", tmp_path / "drift.csv"
    drifted.to_csv(csv, index=False)
    assert drift_check.main([str(csv), "--out", str(out)]) == 1
    report = pd.read_csv(out)
    assert {"Contract", "MonthlyCharges"} <= set(report.loc[report["flag"], "feature"])
    capsys.readouterr()
