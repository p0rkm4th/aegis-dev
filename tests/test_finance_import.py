from __future__ import annotations

import os
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
from aegis.projections import SharedObligation


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


@pytest.mark.parametrize("status", ["pending", "PENDING", "posted", "POSTED"])
def test_csv_import_accepts_only_supported_statuses(status):
    transactions, _ = import_csv_transactions(
        f"date,amount,description,status\n2026-09-01,1.00,Good,{status}\n",
        owner_id="alice",
        account_id="checking",
        source_id="status-test",
    )
    assert transactions[0].status == status.lower()


@pytest.mark.parametrize("status", ["processing", "banana", "settld"])
def test_csv_import_rejects_unknown_status(status):
    transactions, report = import_csv_transactions(
        f"date,amount,description,status\n2026-09-01,1.00,Bad,{status}\n",
        owner_id="alice",
        account_id="checking",
        source_id="status-test",
    )
    assert transactions == ()
    assert report.rejected_rows[0][0] == 2
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


def test_cross_source_similarity_is_candidate_and_pending_provider_id_becomes_posted():
    ledger = FinanceLedger()
    ledger.record_snapshot(
        FinanceSnapshot(
            "alice",
            (Account("checking", "alice", 10_000),),
        )
    )
    base = "date,amount,description\n2026-09-01,-12.34,Coffee\n"
    ledger.import_csv("alice", "checking", base, source_id="upload-a")
    report = ledger.import_csv("alice", "checking", base + "\n", source_id="upload-b")
    assert report.reconciliations[0].classification == "STRONG_CROSS_SOURCE_CANDIDATE"
    assert len(ledger.private_snapshot(type("P", (), {"id": "alice"})(), "alice").transactions) == 2

    pending = (
        "date,amount,description,transaction_id,status\n2026-09-02,-5.00,Coffee,bank-9,pending\n"
    )
    posted = pending.replace(",pending", ",posted")
    ledger.import_csv("alice", "checking", pending, source_id="upload-c")
    ledger.import_csv("alice", "checking", posted, source_id="upload-d")
    rows = ledger.private_snapshot(type("P", (), {"id": "alice"})(), "alice").transactions
    assert len([row for row in rows if row.provider_transaction_id == "bank-9"]) == 1
    assert [row.status for row in rows if row.provider_transaction_id == "bank-9"] == ["posted"]


def test_postgres_finance_round_trip_preserves_non_default_transaction_semantics():
    if not os.environ.get("AEGIS_TEST_DATABASE_URL"):
        pytest.skip("requires disposable PostgreSQL")
    from datetime import timezone

    import psycopg

    from aegis.cli import _apply_migrations
    from aegis.finance import ImportSource, PostgresFinanceSnapshotStore

    url = os.environ["AEGIS_TEST_DATABASE_URL"]
    owner = "finance-fidelity-test"
    captured = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    snapshot = FinanceSnapshot(
        owner,
        (
            Account("euro-checking", owner, 123456, "EUR", captured, "active"),
            Account("usd-checking", owner, 50000, "USD", captured, "active"),
        ),
        (
            Transaction(
                "txn-a",
                "euro-checking",
                -1234,
                captured,
                "Pending purchase",
                "EUR",
                "provider-123",
                "pending",
                "import-a",
            ),
            Transaction(
                "txn-b",
                "usd-checking",
                -500,
                captured,
                "Posted purchase",
                "USD",
                None,
                "posted",
                "import-b",
            ),
        ),
        "bank-evidence",
        captured,
        (
            ImportSource("import-a", "csv", "hash-a", captured, captured, captured, False),
            ImportSource("import-b", "csv", "hash-b", captured, captured, captured, True),
        ),
    )
    setup = psycopg.connect(url)
    try:
        _apply_migrations(setup)
        setup.execute(
            "INSERT INTO aegis_principals (id, external_subject) VALUES (%s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (owner, owner),
        )
        setup.commit()
        PostgresFinanceSnapshotStore(setup).save(snapshot)
    finally:
        setup.close()
    reader = psycopg.connect(url)
    try:
        restored = PostgresFinanceSnapshotStore(reader).load(owner)
        assert restored == snapshot
    finally:
        reader.execute("DELETE FROM finance_snapshots WHERE owner_id = %s", (owner,))
        reader.execute("DELETE FROM aegis_principals WHERE id = %s", (owner,))
        reader.commit()
        reader.close()


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
    assert summary["spend_by_description"] == {
        "USD": {"posted": {"Food": 500}, "pending": {"Hold": 700}}
    }


def test_affordability_is_currency_scoped_and_rejects_mixed_obligations():
    ledger = FinanceLedger()
    ledger.record_snapshot(
        FinanceSnapshot(
            "alice",
            (Account("usd", "alice", 10_000, "USD"), Account("eur", "alice", 99_999, "EUR")),
        )
    )
    principal = type("P", (), {"id": "alice"})()
    projection = ledger.assess_affordability(
        principal,
        "alice",
        8_000,
        (SharedObligation("rent", 1_000, "USD"),),
        purchase_currency="USD",
    )
    assert projection.matching_balance_cents == 10_000
    assert projection.purchase_currency == "USD"
    assert projection.affordable
    assert projection.coverage_complete is None
    with pytest.raises(ValueError, match="purchase currency"):
        ledger.assess_affordability(
            principal,
            "alice",
            8_000,
            (SharedObligation("rent", 1_000, "EUR"),),
            purchase_currency="USD",
        )
