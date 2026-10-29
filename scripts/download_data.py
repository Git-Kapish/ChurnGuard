"""Fetch the IBM Telco churn CSV to data/raw/telco_churn.csv and record its SHA256.

Raw data is read-only: an existing file is never overwritten, only checksummed.
"""

import csv
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"
CHECKSUM_PATH = RAW_PATH.with_name(RAW_PATH.name + ".sha256")
EXPECTED_ROWS, EXPECTED_COLS = 7043, 21
# The file every published number in this repo was built from.
KNOWN_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
TIMEOUT_S = 30

log = logging.getLogger("download_data")

FALLBACK = (
    "Could not obtain the dataset automatically. Download the Kaggle copy "
    "('WA_Fn-UseC_-Telco-Customer-Churn.csv') and save it as "
    f"{RAW_PATH}, then re-run this script to record its checksum."
)


def sha256_of(path: Path) -> str:
    """Return the hex SHA256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shape(path: Path) -> None:
    """Raise ValueError unless the CSV has the expected row and column counts."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        n_rows = sum(1 for _ in reader)
    if len(header) != EXPECTED_COLS or n_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected shape: {n_rows} rows x {len(header)} cols "
            f"(expected {EXPECTED_ROWS} x {EXPECTED_COLS})"
        )


def download(dest: Path) -> None:
    """Download URL to dest via a temp file so a failed fetch never leaves a partial raw file."""
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(URL, timeout=TIMEOUT_S) as resp, tmp.open("wb") as out:
            out.write(resp.read())
        validate_shape(tmp)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    """Ensure the raw CSV and its checksum file exist and agree."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        log.info("Downloading %s", URL)
        try:
            download(RAW_PATH)
        except (OSError, ValueError) as exc:
            log.error("Download failed: %s", exc)
            log.error(FALLBACK)
            return 1
    else:
        try:
            validate_shape(RAW_PATH)
        except (OSError, ValueError) as exc:
            log.error("Existing %s looks wrong: %s", RAW_PATH, exc)
            return 1

    checksum = sha256_of(RAW_PATH)
    if checksum != KNOWN_SHA256:  # e.g. upstream changed, or a re-saved Kaggle copy
        log.warning(
            "WARNING: %s differs from the dataset the published results were built on "
            "(sha256 %s, expected %s). Retrained metrics may not match README/MODEL_CARD.",
            RAW_PATH.name,
            checksum,
            KNOWN_SHA256,
        )
    if CHECKSUM_PATH.exists():
        recorded = CHECKSUM_PATH.read_text().split()[0]
        if recorded != checksum:
            log.error("Checksum mismatch: recorded %s, actual %s", recorded, checksum)
            return 1
        log.info("OK: %s unchanged (sha256 %s)", RAW_PATH.name, checksum)
    else:
        CHECKSUM_PATH.write_text(f"{checksum}  {RAW_PATH.name}\n")
        log.info("Wrote %s (sha256 %s)", CHECKSUM_PATH.name, checksum)
    return 0


if __name__ == "__main__":
    sys.exit(main())
