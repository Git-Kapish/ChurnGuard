from pathlib import Path

import yaml

import churnguard

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_package_imports() -> None:
    assert churnguard.__version__


def test_configs_load() -> None:
    model = yaml.safe_load((CONFIGS / "model.yaml").read_text())
    business = yaml.safe_load((CONFIGS / "business.yaml").read_text())
    assert model["seed"] == 42
    assert round(sum(model["split"].values()), 6) == 1.0
    assert {"retention_offer_cost", "offer_success_rate", "customer_lifetime_value"} <= set(
        business
    )
