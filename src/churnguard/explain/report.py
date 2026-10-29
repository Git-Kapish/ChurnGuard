"""Global explainability: SHAP beeswarm + top-10 bar, and a SHAP vs permutation comparison.

Computed on the validation split with the base (uncalibrated) pipeline. Nothing here reads
the test split.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from churnguard.config import ROOT, load_config  # noqa: E402
from churnguard.data.clean import normalize_customers  # noqa: E402
from churnguard.data.split import load_split, split_xy  # noqa: E402
from churnguard.explain.explainer import Explainer  # noqa: E402
from churnguard.models.artifact import MODEL_PATH, load_bundle  # noqa: E402

log = logging.getLogger("churnguard.explain")
REPORTS_DIR = ROOT / "reports"
BLUE = "#4c78a8"


def permutation_importance_by_group(
    clf: Any,
    Xt: pd.DataFrame,
    y: np.ndarray,
    column_group: dict[str, str],
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.Series:
    """Drop in PR-AUC when a whole attribute group is shuffled, averaged over repeats.

    All columns of a group (e.g. the three Contract one-hot columns plus is_month_to_month)
    are shuffled with the same row permutation so the group stays internally consistent.
    """
    rng = np.random.default_rng(seed)
    baseline = average_precision_score(y, clf.predict_proba(Xt)[:, 1])
    drops: dict[str, float] = {}
    for group in sorted(set(column_group.values())):
        cols = [c for c, g in column_group.items() if g == group]
        scores = []
        for _ in range(n_repeats):
            shuffled = Xt.copy()
            shuffled[cols] = Xt[cols].to_numpy()[rng.permutation(len(Xt))]
            scores.append(average_precision_score(y, clf.predict_proba(shuffled)[:, 1]))
        drops[group] = baseline - float(np.mean(scores))
    return pd.Series(drops).sort_values(ascending=False)


def compare_rankings(shap_imp: pd.Series, perm_imp: pd.Series) -> dict[str, Any]:
    """Rank agreement between mean |SHAP| and permutation importance, per attribute group."""
    ranks = pd.DataFrame(
        {"shap_rank": shap_imp.rank(ascending=False), "perm_rank": perm_imp.rank(ascending=False)}
    )
    rho = float(spearmanr(shap_imp.loc[ranks.index], perm_imp.loc[ranks.index])[0])
    gap = (ranks["shap_rank"] - ranks["perm_rank"]).abs()
    return {"spearman": rho, "ranks": ranks.sort_values("shap_rank"), "big_gaps": gap[gap >= 4]}


def run(
    reports_dir: Path = REPORTS_DIR,
    model_path: Path = MODEL_PATH,
    sample: int | None = None,
) -> dict[str, Any]:
    """Write SHAP figures and the SHAP-vs-permutation note; return the key numbers."""
    seed = load_config("model")["seed"]
    base = load_bundle(model_path)["base"]
    Xtr, _ = split_xy(load_split("train"))
    Xv, yv = split_xy(load_split("val"))
    if sample:
        Xv, yv = Xv.iloc[:sample], yv.iloc[:sample]
    Xv = normalize_customers(Xv)
    background = normalize_customers(Xtr.sample(min(200, len(Xtr)), random_state=seed))
    ex = Explainer(base, background)

    Xt = ex.transform(Xv)
    sv = ex.shap_values(Xv)
    grouped = ex.group_contributions(Xv)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(
        shap.Explanation(sv.to_numpy(), data=Xt.to_numpy(), feature_names=ex.feature_names),
        max_display=15,
        show=False,
    )
    plt.title("SHAP values per model column (validation split, log-odds of churn)")
    plt.gcf().savefig(fig_dir / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    shap_imp = grouped.abs().mean().sort_values(ascending=False)
    top = shap_imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.to_numpy(), color=BLUE)
    ax.set(
        xlabel="mean |SHAP| (log-odds), columns grouped by customer attribute",
        title="Top 10 drivers of predicted churn (validation split)",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "shap_top10_bar.png", dpi=130)
    plt.close(fig)

    perm = permutation_importance_by_group(ex.clf, Xt, yv.to_numpy(), ex.column_group, seed=seed)
    cmp = compare_rankings(shap_imp, perm)
    note = render_note(shap_imp, perm, cmp, len(Xv))
    (reports_dir / "explainability_notes.md").write_text(note, encoding="utf-8")
    log.info("SHAP vs permutation rank correlation %.2f", cmp["spearman"])
    return {"spearman": cmp["spearman"], "top_shap": list(shap_imp.head(10).index), "n": len(Xv)}


def render_note(shap_imp: pd.Series, perm: pd.Series, cmp: dict[str, Any], n: int) -> str:
    """Markdown note: both rankings, their agreement and the disagreements."""
    ranks = cmp["ranks"]
    lines = [
        "# Explainability notes",
        "",
        f"Computed on the validation split ({n} customers) with the base (uncalibrated) XGBoost "
        "pipeline. SHAP values are in log-odds of the base model; the calibrated model supplies "
        "the probabilities shown to users. Calibration is a monotone map, so the direction of a "
        "factor is preserved, but SHAP magnitudes are not calibrated probability points.",
        "",
        "Columns are grouped by customer attribute (all Contract columns together, and so on) "
        "for both methods, so the rankings compare like with like.",
        "",
        f"Spearman rank correlation between mean |SHAP| and permutation importance: "
        f"**{cmp['spearman']:.2f}**.",
        "",
        "| Attribute | Mean abs SHAP | SHAP rank | PR-AUC drop when shuffled | Permutation rank |",
        "|---|---|---|---|---|",
    ]
    for g in ranks.index:
        lines.append(
            f"| {g} | {shap_imp[g]:.3f} | {int(ranks.loc[g, 'shap_rank'])} | "
            f"{perm[g]:+.4f} | {int(ranks.loc[g, 'perm_rank'])} |"
        )
    lines += ["", "## Disagreements", ""]
    if cmp["big_gaps"].empty:
        lines.append("No attribute differs by 4 or more ranks between the two methods.")
    else:
        for g, gap in cmp["big_gaps"].items():
            lines.append(
                f"- **{g}**: SHAP rank {int(ranks.loc[g, 'shap_rank'])}, permutation rank "
                f"{int(ranks.loc[g, 'perm_rank'])} (gap {int(gap)})."
            )
    top_shap = set(shap_imp.head(5).index)
    top_perm = set(perm.head(5).index)
    negative = [g for g in perm.index if perm[g] < 0]
    lines += ["", "## Reading the comparison", ""]
    if top_shap == top_perm:
        lines.append("- The same five attributes lead both rankings, so the main drivers agree.")
    else:
        lines.append(
            f"- The top five differ: only in SHAP: {sorted(top_shap - top_perm)}; "
            f"only in permutation: {sorted(top_perm - top_shap)}."
        )
    if negative:
        lines.append(
            f"- Shuffling {', '.join(negative)} slightly *improved* held-out PR-AUC. The model "
            "uses these attributes (non-zero SHAP) but they add no reliable signal on unseen "
            "customers. Drops this small are also within permutation noise, so read them as "
            "'no evidence of value', not as proof of harm."
        )
    lines += [
        "- SHAP measures how much an attribute moves the model's own predictions; permutation "
        "importance measures how much held-out PR-AUC falls when it is scrambled. Correlated "
        "attributes can rank differently under the two, and shuffling one attribute can create "
        "unrealistic combinations that inflate its permutation score.",
        "- Both describe this model, not causal effects on customer behavior.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entrypoint: `python -m churnguard.explain.report`."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()


if __name__ == "__main__":
    main()
