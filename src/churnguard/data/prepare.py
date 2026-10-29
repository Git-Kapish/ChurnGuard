"""CLI: validate raw data, clean it, split it and write parquet (used by `make data`)."""

import logging

from churnguard.config import load_config
from churnguard.data.clean import clean
from churnguard.data.load import load_raw
from churnguard.data.split import save_splits, split_data
from churnguard.data.validate import validate_clean, validate_raw

log = logging.getLogger("churnguard.data")


def main() -> None:
    """Run load -> validate -> clean -> validate -> split -> save."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config("model")
    df = validate_clean(clean(validate_raw(load_raw())))
    splits = split_data(df, seed=cfg["seed"], **cfg["split"])
    save_splits(splits)
    for name, frame in splits.items():
        log.info("%-5s %5d rows, churn rate %.3f", name, len(frame), frame["Churn"].mean())


if __name__ == "__main__":
    main()
