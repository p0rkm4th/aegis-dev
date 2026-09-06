from __future__ import annotations

from datetime import datetime

import pytest

from aegis.finance import (
    Account,
    FinanceLedger,
    FinanceSnapshot,
    Transaction,
    import_csv_transactions,
    summarize_snapshot,
)


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


def test_ledger_import_merges_once_and_preserves_prior_snapshot():
    ledger = FinanceLedger()
    ledger.record_snapshot(FinanceSnapshot("alice", (Account("checking", "alice", 10_000),)))
    content = "date,amount,description,transaction_id\n2026-09-01,-12.34,Coffee,bank-1\n"

    first = ledger.import_csv("alice", "checking", content, source_id="upload-1")
    second = ledger.import_csv("alice", "checking", content, source_id="upload-1")
    snapshot = ledger.private_snapshot(type("Principal", (), {"id": "alice"})(), "alice")

    assert first.imported_transaction_ids == second.imported_transaction_ids
    assert len(snapshot.transactions) == 1
    assert len(snapshot.sources) == 1


def test_ledger_import_does_not_double_count_provider_transaction_across_files():
    ledger = FinanceLedger()
    ledger.record_snapshot(FinanceSnapshot("alice", (Account("checking", "alice", 10_000),)))
    row = "date,amount,description,transaction_id\n2026-09-01,-12.34,Coffee,bank-1\n"

    ledger.import_csv("alice", "checking", row, source_id="upload-1")
    ledger.import_csv(
        "alice", "checking", row.replace("Coffee", "Coffee shop"), source_id="upload-2"
    )
    snapshot = ledger.private_snapshot(type("Principal", (), {"id": "alice"})(), "alice")

    assert len(snapshot.transactions) == 1
    assert len(snapshot.sources) == 2


def test_finance_summary_keeps_currencies_and_pending_flow_separate():
    snapshot = FinanceSnapshot(
        "alice",
        (
            Account("usd", "alice", 10_000, currency="USD"),
            Account("eur", "alice", 8_000, currency="EUR"),
        ),
        (
            Transaction("posted", "usd", -500, datetime(2026, 9, 1), "Food"),
            Transaction("pending", "usd", -700, datetime(2026, 9, 2), "Hold", status="pending"),
        ),
    )

    summary = summarize_snapshot(snapshot)
    assert summary["balances_by_currency"] == {"USD": 10_000, "EUR": 8_000}
    assert summary["cash_flow_by_currency"] == {"USD": {"posted": -500, "pending": -700}}
