"""Compare a new customers file against the training data and flag features with PSI > 0.2.

    python scripts/drift_check.py data/processed/val.parquet
    python scripts/drift_check.py new_customers.csv --out drift.csv

Accepts .csv or .parquet with the Telco customer columns. Exit code 1 if any feature is flagged.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from churnguard.data.drift import THRESHOLD, drift_report
from churnguard.data.split import load_split

log = logging.getLogger("drift_check")


def read_customers(path: Path) -> pd.DataFrame:
    """Read a customers file. CSV is read as text (blank TotalCharges are coerced later by
    normalize_customers); the other numeric columns are converted here."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("tenure", "SeniorCitizen", "MonthlyCharges"):
        df[col] = pd.to_numeric(df[col])
    return df


def main(argv: list[str] | None = None) -> int:
    """Run the drift check; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_data", type=Path, help="CSV or parquet of new customers")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="PSI flag threshold")
    parser.add_argument("--out", type=Path, help="also write the report as CSV")
    args = parser.parse_args(argv)

    report = drift_report(load_split("train"), read_customers(args.new_data), args.threshold)
    log.info(
        "PSI per feature (training data vs %s)\n%s", args.new_data, report.to_string(index=False)
    )
    if args.out:
        report.to_csv(args.out, index=False)
    flagged = report[report["flag"]]
    if flagged.empty:
        log.info("No feature drifted beyond PSI %.2f.", args.threshold)
        return 0
    log.warning(
        "DRIFT: %d feature(s) above PSI %.2f: %s",
        len(flagged),
        args.threshold,
        ", ".join(flagged["feature"]),
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
