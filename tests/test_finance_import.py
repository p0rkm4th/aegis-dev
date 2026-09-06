from __future__ import annotations

import pytest

from aegis.finance import import_csv_transactions


def test_csv_import_uses_minor_units_source_hash_and_deduplicates_rows():
    content = (
        "date,amount,description,transaction_id\n"
        "2026-09-01,-12.34,Coffee,bank-1\n"
        "2026-09-01,-12.34,Coffee,bank-1\n"
        "2026-09-02,5.00,Refund,bank-2\n"
    )
    transactions, report = import_csv_transactions(
        content,
        owner_id="alice",
        account_id="checking",
        source_id="upload-1",
    )

    assert [item.amount_cents for item in transactions] == [-1234, 500]
    assert transactions[0].currency == "USD"
    assert transactions[0].provider_transaction_id == "bank-1"
    assert transactions[0].source_id == "upload-1"
    assert report.duplicate_rows == (3,)
    assert report.rejected_rows == ()
    assert len(report.source.content_hash) == 64
    assert report.source.coverage_start is not None
    assert report.source.coverage_end is not None


def test_csv_import_keeps_bad_rows_rejected_and_marks_coverage_partial():
    transactions, report = import_csv_transactions(
        "date,amount,description\n2026-09-01,1.10,Good\nnot-a-date,nope,Bad\n",
        owner_id="alice",
        account_id="checking",
        source_id="upload-2",
    )

    assert len(transactions) == 1
    assert report.source.complete is False
    assert report.rejected_rows[0][0] == 3


def test_csv_import_rejects_missing_schema_and_invalid_currency():
    with pytest.raises(ValueError, match="requires date"):
        import_csv_transactions(
            "date,amount\n2026-09-01,1.00\n",
            owner_id="alice",
            account_id="checking",
            source_id="upload-3",
        )
    with pytest.raises(ValueError, match="currency"):
        import_csv_transactions(
            "date,amount,description\n2026-09-01,1.00,Good\n",
            owner_id="alice",
            account_id="checking",
            source_id="upload-4",
            currency="US",
        )
