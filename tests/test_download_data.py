import logging

import pytest

from churnguard.config import RAW_PATH, ROOT


@pytest.fixture
def download_data(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import download_data

    return download_data


def fake_csv(path, rows: int = 7043, cols: int = 21) -> None:
    header = ",".join(f"c{i}" for i in range(cols))
    path.write_text(header + "\n" + ("\n".join(",".join("1" * cols) for _ in range(rows))) + "\n")


def test_unknown_file_is_accepted_but_flagged(download_data, tmp_path, monkeypatch, caplog) -> None:
    raw = tmp_path / "telco_churn.csv"
    fake_csv(raw)
    monkeypatch.setattr(download_data, "RAW_PATH", raw)
    monkeypatch.setattr(download_data, "CHECKSUM_PATH", tmp_path / "telco_churn.csv.sha256")
    with caplog.at_level(logging.WARNING, logger="download_data"):
        assert download_data.main() == 0
    assert any("differs from the dataset" in r.message for r in caplog.records)
    assert (tmp_path / "telco_churn.csv.sha256").exists()


def test_wrong_shape_is_refused(download_data, tmp_path, monkeypatch) -> None:
    raw = tmp_path / "telco_churn.csv"
    fake_csv(raw, rows=10)
    monkeypatch.setattr(download_data, "RAW_PATH", raw)
    monkeypatch.setattr(download_data, "CHECKSUM_PATH", tmp_path / "telco_churn.csv.sha256")
    assert download_data.main() == 1


@pytest.mark.skipif(not RAW_PATH.exists(), reason="run `make data` first")
def test_the_real_dataset_matches_the_published_checksum(download_data) -> None:
    assert download_data.sha256_of(RAW_PATH) == download_data.KNOWN_SHA256
