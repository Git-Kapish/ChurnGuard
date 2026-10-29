"""Run each dashboard page headlessly (Streamlit AppTest) against the real model."""

import pytest
from streamlit.testing.v1 import AppTest

from churnguard.config import PROCESSED_DIR, ROOT
from churnguard.models.artifact import MODEL_PATH

APP_DIR = ROOT / "app"
pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and (PROCESSED_DIR / "val.parquet").exists()),
    reason="run `make data train evaluate` first",
)


def run_page(module: str) -> AppTest:
    script = (
        f"import sys; sys.path.insert(0, {str(APP_DIR)!r})\n"
        f"from views import {module}\n{module}.render()\n"
    )
    return AppTest.from_string(script, default_timeout=120).run()


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_entrypoint_builds_four_pages() -> None:
    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Overview"  # default page renders


def test_overview_renders_and_the_filter_works() -> None:
    at = run_page("overview")
    assert not at.exception
    assert at.title[0].value == "Overview"
    assert metric(at, "Customers") == "4,930"
    assert len(at.tabs) == 4
    at.multiselect[0].set_value(["Two year"]).run()
    assert not at.exception
    assert metric(at, "Customers") == "1,192"  # two-year contracts in the training split
    at.multiselect[0].set_value([]).run()
    assert at.warning and "No customers" in at.warning[0].value


def test_overview_explore_dimension_changes_the_table() -> None:
    at = run_page("overview")
    at.selectbox[0].set_value("Senior citizen").run()
    assert not at.exception
    assert set(at.dataframe[0].value["Senior citizen"]) == {"No", "Yes"}


def test_model_page_shows_test_results_and_targets() -> None:
    at = run_page("model")
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"PR-AUC (primary)", "ROC-AUC", "Brier score"} <= set(labels)
    assert any("Met" in df.value.columns for df in at.dataframe)
    targets = next(df.value for df in at.dataframe if "Met" in df.value.columns)
    money = targets[targets["Target"].str.contains("net_value")].iloc[0]
    assert "assumption-driven" in money["Target"]
    assert money["Result"].startswith("+") and "." not in money["Result"]  # whole-number amount
    assert any("assumption" in c.value.lower() for c in at.caption)


def test_score_page_asks_before_scoring_then_scores_each_preset() -> None:
    at = run_page("score")
    assert not at.exception and at.info
    expectations = {"High-risk example": "High", "Low-risk example": "Low"}
    for preset, tier in expectations.items():
        at.radio[0].set_value(preset).run()
        at.button[0].click().run()
        assert not at.exception, at.exception
        assert metric(at, "Risk tier") == tier
        assert len(at.get("plotly_chart")) == 2  # gauge + waterfall
        assert any("▲" in m.value or "▼" in m.value for m in at.markdown)  # reasons listed
        dollars = [m.value for m in at.markdown if "Total charges of" in m.value]
        assert all("\\$" in v for v in dollars)  # escaped, so Streamlit does not render it as math


def test_score_page_probability_matches_predict_py(monkeypatch) -> None:
    from churnguard.api.schemas import Customer
    from churnguard.predict import predict

    monkeypatch.syspath_prepend(str(APP_DIR))
    from views.score import HIGH_RISK

    at = run_page("score")
    at.radio[0].set_value("High-risk example").run()
    at.button[0].click().run()
    expected = predict([Customer.model_validate(HIGH_RISK).model_dump(mode="json")])[0]
    shown = at.session_state["scored"]["result"]
    assert shown["churn_probability"] == pytest.approx(expected["churn_probability"])
    assert shown["risk_tier"] == expected["risk_tier"]


def test_score_page_total_charges_checkbox_allows_a_new_customer() -> None:
    at = run_page("score")
    at.checkbox[0].check()
    at.number_input[0].set_value(0)  # tenure 0
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert metric(at, "Risk tier") in {"Low", "Medium", "High"}


def test_simulator_defaults_and_budget_slider() -> None:
    at = run_page("simulator")
    assert not at.exception
    assert any("assumption" in w.value.lower() for w in at.warning)
    default_targets = int(metric(at, "Customers to contact").replace(",", ""))
    assert 0 < default_targets <= 150  # 3000 budget / 20 offer cost

    at.slider[0].set_value(500).run()  # budget 500 -> 25 offers
    assert int(metric(at, "Customers to contact")) == 25
    assert metric(at, "Spend") == "500"

    at.slider[0].set_value(0).run()
    assert int(metric(at, "Customers to contact")) == 0
    assert at.get("download_button")[0].proto.disabled is True


def test_simulator_expensive_offers_shrink_the_list() -> None:
    at = run_page("simulator")
    base = int(metric(at, "Customers to contact").replace(",", ""))
    at.slider[1].set_value(100).run()  # offer cost 100 -> break-even 66.7% and 30 offers max
    assert not at.exception
    assert int(metric(at, "Customers to contact").replace(",", "")) < base


def test_simulator_shows_outcomes_for_the_validation_pool() -> None:
    at = run_page("simulator")
    labels = [m.label for m in at.metric]
    assert "Net value using actual outcomes" in labels
    assert at.dataframe  # strategy table and target list
    assert any("Validation customers were used" in c.value for c in at.caption)


def test_simulator_upload_mode_waits_for_a_file() -> None:
    # AppTest cannot upload files; CSV validation itself is covered in test_csv_input.py
    at = run_page("simulator")
    at.radio[0].set_value("Upload a CSV of customers").run()
    assert not at.exception
    assert any("Upload a CSV" in i.value for i in at.info)
