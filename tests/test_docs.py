"""Keep the docs honest: published numbers must match the saved results, packaging must hold."""

import json
import re

import pytest
import yaml

from churnguard.config import ROOT

README = ROOT / "README.md"
CARD = ROOT / "reports" / "MODEL_CARD.md"
RESULTS = ROOT / "reports" / "test_metrics.json"
needs_results = pytest.mark.skipif(not RESULTS.exists(), reason="run `make evaluate` first")


@needs_results
@pytest.mark.parametrize("doc", [README, CARD], ids=lambda p: p.name)
def test_docs_quote_the_real_test_metrics(doc) -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    metrics, ci = results["metrics"], results["bootstrap_95ci"]
    text = doc.read_text(encoding="utf-8")
    for key in ("pr_auc", "roc_auc", "brier"):
        assert f"{metrics[key]:.3f}" in text, f"{doc.name} is missing {key}"
        lo, hi = ci[key]
        assert f"[{lo:.3f}, {hi:.3f}]" in text, f"{doc.name} is missing the {key} interval"
    assert f"{metrics['recall']:.3f}" in text and f"{metrics['top_decile_lift']:.2f}" in text
    assert f"{metrics['threshold']:.3f}" in text


@needs_results
def test_docs_quote_the_real_net_value_comparison() -> None:
    strategies = json.loads(RESULTS.read_text(encoding="utf-8"))["strategies"]
    chosen = strategies["chosen_threshold"]["net_value"]
    everyone = strategies["target_everyone"]["net_value"]
    for doc in (README, CARD):
        text = doc.read_text(encoding="utf-8")
        assert f"{chosen:,.0f}" in text and f"{everyone:,.0f}" in text
        assert f"{chosen - everyone:,.0f}" in text


def test_docs_label_money_figures_as_assumptions() -> None:
    for doc in (README, CARD):
        assert "assumption-driven" in doc.read_text(encoding="utf-8").lower()


def test_readme_local_links_and_images_exist() -> None:
    text = README.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    local = [t.split("#")[0] for t in targets if not t.startswith(("http", "#"))]
    assert local, "expected relative links"
    missing = [t for t in local if t and not (ROOT / t).exists()]
    assert not missing, f"broken README links: {missing}"


def test_packaging_files_exist() -> None:
    for name in (
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        ".github/workflows/ci.yml",
        "requirements.lock",
        "scripts/drift_check.py",
        "reports/MODEL_CARD.md",
    ):
        assert (ROOT / name).exists(), name


def test_dockerfile_is_multistage_and_non_root() -> None:
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    stages = [ln for ln in lines if ln.startswith("FROM ")]
    assert len(stages) >= 2 and all(" AS " in s for s in stages)
    assert any(ln.strip() == "USER app" for ln in lines)
    assert any("requirements.lock" in ln for ln in lines)  # pinned install
    assert not any(ln.startswith("COPY") and "data" in ln.split()[1:2] for ln in lines)


def test_compose_runs_api_and_dashboard_on_the_documented_ports() -> None:
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]
    assert set(services) == {"api", "app"}
    assert "8000:8000" in services["api"]["ports"] and "8501:8501" in services["app"]["ports"]
    assert services["api"]["image"] == "churnguard:local"
    assert services["app"]["image"] == services["api"]["image"]  # one build, two services
    assert any(v.startswith("./data:") for v in services["app"]["volumes"])


def test_lock_pins_everything_and_omits_training_only_packages() -> None:
    entries = [
        ln.strip()
        for ln in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith(("#", "--"))
    ]
    pinned = [e for e in entries if "==" in e]
    names = {e.split("==")[0].lower() for e in pinned}
    assert len(pinned) > 30
    assert {"xgboost", "scikit-learn", "shap", "fastapi", "streamlit", "pydantic"} <= {
        n.removesuffix("-cpu") for n in names
    }
    assert not names & {"mlflow", "optuna", "pytest", "ruff", "black"}


def test_ci_runs_lint_and_tests_with_a_coverage_floor() -> None:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = ci["jobs"]["lint-and-test"]["steps"]
    commands = " ".join(s.get("run", "") for s in steps)
    assert "ruff check" in commands and "black --check" in commands
    assert "pytest" in commands and "--cov-fail-under=80" in commands
