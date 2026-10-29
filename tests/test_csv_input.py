import pandas as pd
import pytest

from churnguard.api.csv_input import MAX_CSV_BYTES, CsvInputError, parse_customers_csv


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def test_valid_csv_returns_the_frame_and_clean_records(random_customers) -> None:
    df = random_customers(n=5).drop(columns=["customerID"])
    frame, records = parse_customers_csv(csv_bytes(df))
    assert len(frame) == 5 and len(records) == 5
    assert records[0]["Contract"] in {"Month-to-month", "One year", "Two year"}
    assert isinstance(records[0]["tenure"], int) and isinstance(records[0]["MonthlyCharges"], float)
    assert "Churn" in frame.columns and "Churn" not in records[0]  # extra column kept, not scored


def test_blank_optional_cells_become_none(random_customers) -> None:
    df = random_customers(n=3).astype({"TotalCharges": object})
    df.loc[0, "TotalCharges"] = " "
    _, records = parse_customers_csv(csv_bytes(df))
    assert records[0]["TotalCharges"] is None and records[0]["customerID"] == "C0"


@pytest.mark.parametrize(
    ("kind", "status"),
    [("empty", 400), ("header only", 422), ("wrong columns", 422), ("too large", 413)],
)
def test_unusable_csvs_raise_with_the_right_status(kind: str, status: int) -> None:
    raw = {
        "empty": b"",
        "header only": b"a,b\n",
        "wrong columns": b"a,b\n1,2\n",
        "too large": b"x" * (MAX_CSV_BYTES + 1),
    }[kind]
    with pytest.raises(CsvInputError) as err:
        parse_customers_csv(raw)
    assert err.value.status_code == status


def test_errors_name_the_line_and_are_capped(random_customers) -> None:
    df = random_customers(n=12).drop(columns=["customerID"])
    df.loc[df.index[:8], "Contract"] = "Weekly"
    with pytest.raises(CsvInputError) as err:
        parse_customers_csv(csv_bytes(df))
    detail = err.value.detail
    assert detail[0].startswith("row 2: Contract") and len(detail) == 6
    assert detail[-1] == "... 3 more"
