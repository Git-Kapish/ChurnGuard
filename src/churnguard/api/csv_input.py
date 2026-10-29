"""Validate an uploaded customers CSV. Shared by the API (/predict/csv) and the dashboard."""

import io

import pandas as pd
from pydantic import ValidationError

from churnguard.api.schemas import MAX_BATCH, Customer

MAX_CSV_BYTES = 2_000_000  # ~7x what 1,000 customers need; rejects anything absurd early
MAX_ERRORS_SHOWN = 5
OPTIONAL_COLUMNS = {"customerID", "TotalCharges"}  # blank cells mean "not provided"


class CsvInputError(Exception):
    """A CSV the caller should fix. `status_code` is the matching HTTP status."""

    def __init__(self, status_code: int, detail: str | list[str]) -> None:
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


def parse_customers_csv(raw: bytes) -> tuple[pd.DataFrame, list[dict]]:
    """Return (the CSV as read, validated customer records) or raise CsvInputError.

    Extra columns (e.g. Churn) are kept in the frame but ignored for scoring. Errors name the
    CSV line (header = line 1) of the first problem in each bad row.
    """
    if len(raw) > MAX_CSV_BYTES:
        raise CsvInputError(413, f"CSV larger than {MAX_CSV_BYTES} bytes")
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise CsvInputError(400, f"could not read CSV: {exc}") from exc
    if df.empty:
        raise CsvInputError(422, "CSV has no rows")
    if len(df) > MAX_BATCH:
        raise CsvInputError(422, f"at most {MAX_BATCH} rows per CSV")
    fields = list(Customer.model_fields)
    missing = [f for f in fields if f not in df.columns and f not in OPTIONAL_COLUMNS]
    if missing:
        raise CsvInputError(422, f"missing columns: {missing}")

    records, errors = [], []
    # absent optional columns become blank cells, which are then treated as "not provided"
    for i, row in enumerate(df.reindex(columns=fields, fill_value="").to_dict("records")):
        row = {k: v for k, v in row.items() if not (k in OPTIONAL_COLUMNS and not str(v).strip())}
        try:
            records.append(Customer.model_validate(row).model_dump(mode="json"))
        except ValidationError as exc:
            first = exc.errors()[0]
            errors.append(f"row {i + 2}: {'.'.join(map(str, first['loc']))}: {first['msg']}")
    if errors:
        shown = errors[:MAX_ERRORS_SHOWN]
        more = len(errors) - len(shown)
        raise CsvInputError(422, shown + ([f"... {more} more"] if more else []))
    return df, records
