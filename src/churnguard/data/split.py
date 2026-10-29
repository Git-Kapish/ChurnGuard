"""Stratified train/validation/test split and parquet persistence."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard.config import PROCESSED_DIR


def split_data(
    df: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    target: str = "Churn",
) -> dict[str, pd.DataFrame]:
    """Stratified split into train/val/test by the given fractions (must sum to 1)."""
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    train_df, rest = train_test_split(df, train_size=train, stratify=df[target], random_state=seed)
    val_df, test_df = train_test_split(
        rest, train_size=val / (val + test), stratify=rest[target], random_state=seed
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def split_xy(df: pd.DataFrame, target: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, target); customerID is dropped, the model never sees it."""
    return df.drop(columns=[target, "customerID"], errors="ignore"), df[target]


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path = PROCESSED_DIR) -> None:
    """Write each split to <out_dir>/<name>.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", index=False)


def load_split(name: str, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Read a saved split ('train', 'val' or 'test')."""
    return pd.read_parquet(out_dir / f"{name}.parquet")
