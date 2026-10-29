"""Optuna hyperparameter search, optimizing mean CV PR-AUC on the training folds."""

import logging
from typing import Any

import mlflow
import optuna
import pandas as pd

from churnguard.models import tracking
from churnguard.models.cv import Folds, cross_validate, summarize
from churnguard.models.registry import build_estimator

log = logging.getLogger("churnguard.tune")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Sample hyperparameters for model `name`."""
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 0.8),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    raise ValueError(f"no search space for {name}")


def tune(
    name: str,
    imbalance: str,
    X: pd.DataFrame,
    y: pd.Series,
    folds: Folds,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Run an Optuna study; each trial is a nested MLflow run.

    Returns {"best_params", "summary"} where summary is the CV mean/std of the best trial.
    The score is optimistically biased (selected on these folds); Block 4 compares the tuned
    models on the untouched validation split.
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest(name, trial)
        scores = cross_validate(build_estimator(name, imbalance, params, seed), X, y, folds)
        summary = summarize(scores)
        trial.set_user_attr("summary", summary)
        with tracking.run(f"{name}-trial-{trial.number}", params, nested=True):
            mlflow.log_metrics(summary)
        return summary["pr_auc_mean"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    with tracking.run(
        f"{name}-optuna", {"model": name, "imbalance": imbalance, "n_trials": n_trials}
    ):
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics(
            {f"best_{k}": v for k, v in study.best_trial.user_attrs["summary"].items()}
        )
    log.info("%s tuned: best CV PR-AUC %.4f", name, study.best_value)
    return {
        "best_params": {**suggest_fixed(name), **study.best_params},
        "summary": study.best_trial.user_attrs["summary"],
    }


def suggest_fixed(name: str) -> dict[str, Any]:
    """Non-tuned constants that are part of a search space (e.g. LightGBM subsample_freq)."""
    return {"subsample_freq": 1} if name == "lightgbm" else {}
