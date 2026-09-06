"""Provider-independent private finance state and controlled derived outputs."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, Protocol, cast
from uuid import uuid4

from .contracts import IntentFrame, ObjectiveState, Principal, Result
from .projections import PrivateContribution, SharedObligation
from .utterance import has_multiple_question_clauses


@dataclass(frozen=True)
class Account:
    account_id: str
    owner_id: str
    balance_cents: int
    currency: str = "USD"
    balance_as_of: datetime | None = None
    status: str = "active"


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    account_id: str
    amount_cents: int
    occurred_at: datetime
    description: str
    currency: str = "USD"
    provider_transaction_id: str | None = None
    status: str = "posted"
    source_id: str = "fixture"


@dataclass(frozen=True)
class ImportSource:
    source_id: str
    source_type: str
    content_hash: str
    imported_at: datetime
    coverage_start: datetime | None = None
    coverage_end: datetime | None = None
    complete: bool = True


@dataclass(frozen=True)
class FinanceImportReport:
    source: ImportSource
    imported_transaction_ids: tuple[str, ...]
    duplicate_rows: tuple[int, ...] = ()
    rejected_rows: tuple[tuple[int, str], ...] = ()


def import_csv_transactions(
    content: str,
    *,
    owner_id: str,
    account_id: str,
    source_id: str,
    currency: str = "USD",
    imported_at: datetime | None = None,
) -> tuple[tuple[Transaction, ...], FinanceImportReport]:
    """Parse a bounded owner-controlled CSV without floating-point money."""

    if not owner_id or not account_id or not source_id:
        raise ValueError("owner, account, and source identities are required")
    if len(content.encode()) > 5_000_000:
        raise ValueError("finance import exceeds size limit")
    currency = currency.strip().upper()
    if len(currency) != 3 or not currency.isalpha():
        raise ValueError("currency must be an ISO-like three-letter code")
    digest = sha256(content.encode()).hexdigest()
    now = imported_at or datetime.now(timezone.utc)
    reader = csv.DictReader(io.StringIO(content))
    required = {"date", "amount", "description"}
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        raise ValueError("CSV requires date, amount, and description columns")
    transactions: list[Transaction] = []
    seen: set[str] = set()
    duplicates: list[int] = []
    rejected: list[tuple[int, str]] = []
    dates: list[datetime] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            occurred_at = datetime.fromisoformat(str(row.get("date", "")).strip())
            amount = Decimal(str(row.get("amount", "")).strip()).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_EVEN
            )
            if not amount.is_finite():
                raise ValueError("amount is not finite")
            description = str(row.get("description", "")).strip()
            if not description:
                raise ValueError("description is required")
            provider_id = str(row.get("transaction_id", "")).strip() or None
            identity = (
                provider_id
                or sha256(
                    f"{source_id}\0{occurred_at.isoformat()}\0{amount}\0{description}".encode()
                ).hexdigest()
            )
            if identity in seen:
                duplicates.append(row_number)
                continue
            seen.add(identity)
            transaction_id = f"{source_id}:{identity}"
            dates.append(occurred_at)
            transactions.append(
                Transaction(
                    transaction_id=transaction_id,
                    account_id=account_id,
                    amount_cents=int(amount * 100),
                    occurred_at=occurred_at,
                    description=description,
                    currency=currency,
                    provider_transaction_id=provider_id,
                    status=str(row.get("status", "posted")).strip().lower() or "posted",
                    source_id=source_id,
                )
            )
        except (InvalidOperation, TypeError, ValueError) as exc:
            rejected.append((row_number, str(exc)))
    source = ImportSource(
        source_id=source_id,
        source_type="csv",
        content_hash=digest,
        imported_at=now,
        coverage_start=min(dates) if dates else None,
        coverage_end=max(dates) if dates else None,
        complete=not rejected,
    )
    return tuple(transactions), FinanceImportReport(
        source=source,
        imported_transaction_ids=tuple(item.transaction_id for item in transactions),
        duplicate_rows=tuple(duplicates),
        rejected_rows=tuple(rejected),
    )


@dataclass(frozen=True)
class FinanceSnapshot:
    owner_id: str
    accounts: tuple[Account, ...]
    transactions: tuple[Transaction, ...] = ()
    provider_id: str = "fixture"
    captured_at: datetime | None = None
    sources: tuple[ImportSource, ...] = ()


@dataclass(frozen=True)
class AffordabilityProjection:
    purchase_cents: int
    shared_obligations_cents: int
    affordable: bool
    shortfall_cents: int


def summarize_snapshot(snapshot: FinanceSnapshot) -> dict[str, object]:
    """Derive bounded finance summaries without aggregating currencies."""

    balances: dict[str, int] = {}
    for account in snapshot.accounts:
        currency = account.currency.upper()
        balances[currency] = balances.get(currency, 0) + account.balance_cents
    flows: dict[str, dict[str, int]] = {}
    for transaction in snapshot.transactions:
        currency = transaction.currency.upper()
        bucket = flows.setdefault(currency, {"posted": 0, "pending": 0})
        status = "pending" if transaction.status == "pending" else "posted"
        bucket[status] += transaction.amount_cents
    return {
        "balances_by_currency": balances,
        "cash_flow_by_currency": flows,
        "transaction_count": len(snapshot.transactions),
        "source_count": len(snapshot.sources),
        "coverage": [
            {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "content_hash": source.content_hash,
                "coverage_start": source.coverage_start.isoformat()
                if source.coverage_start
                else None,
                "coverage_end": source.coverage_end.isoformat() if source.coverage_end else None,
                "complete": source.complete,
            }
            for source in snapshot.sources
        ],
    }


class FinancialProvider(Protocol):
    def snapshot(self, owner_id: str) -> FinanceSnapshot: ...


class FinanceSnapshotStore(Protocol):
    def save(self, snapshot: FinanceSnapshot) -> None: ...

    def load(self, owner_id: str) -> FinanceSnapshot | None: ...


class PostgresFinanceSnapshotStore:
    """Persist private finance snapshots partitioned by owning principal."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def save(self, snapshot: FinanceSnapshot) -> None:
        if not snapshot.owner_id:
            raise ValueError("finance owner is required")
        payload = {
            "accounts": [
                {
                    "account_id": account.account_id,
                    "owner_id": account.owner_id,
                    "balance_cents": account.balance_cents,
                    "currency": account.currency,
                    "balance_as_of": account.balance_as_of.isoformat()
                    if account.balance_as_of
                    else None,
                    "status": account.status,
                }
                for account in snapshot.accounts
            ],
            "transactions": [
                {
                    "transaction_id": transaction.transaction_id,
                    "account_id": transaction.account_id,
                    "amount_cents": transaction.amount_cents,
                    "occurred_at": transaction.occurred_at.isoformat(),
                    "description": transaction.description,
                }
                for transaction in snapshot.transactions
            ],
            "sources": [
                {
                    "source_id": source.source_id,
                    "source_type": source.source_type,
                    "content_hash": source.content_hash,
                    "imported_at": source.imported_at.isoformat(),
                    "coverage_start": source.coverage_start.isoformat()
                    if source.coverage_start
                    else None,
                    "coverage_end": source.coverage_end.isoformat()
                    if source.coverage_end
                    else None,
                    "complete": source.complete,
                }
                for source in snapshot.sources
            ],
        }
        self.connection.execute(
            "INSERT INTO finance_snapshots "
            "(owner_id, payload, provider_id, captured_at) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (owner_id) DO UPDATE SET payload = EXCLUDED.payload, "
            "provider_id = EXCLUDED.provider_id, captured_at = EXCLUDED.captured_at, "
            "updated_at = now()",
            (
                snapshot.owner_id,
                json.dumps(payload, sort_keys=True),
                snapshot.provider_id,
                snapshot.captured_at,
            ),
        )
        self.connection.commit()

    def load(self, owner_id: str) -> FinanceSnapshot | None:
        row = self.connection.execute(
            "SELECT payload, provider_id, captured_at FROM finance_snapshots WHERE owner_id = %s",
            (owner_id,),
        ).fetchone()
        if row is None:
            return None
        payload = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
        accounts = tuple(
            Account(
                str(account["account_id"]),
                str(account["owner_id"]),
                int(account["balance_cents"]),
                str(account.get("currency", "USD")),
                datetime.fromisoformat(str(account["balance_as_of"]))
                if account.get("balance_as_of")
                else None,
                str(account.get("status", "active")),
            )
            for account in payload.get("accounts", [])
        )
        transactions = tuple(
            Transaction(
                str(transaction["transaction_id"]),
                str(transaction["account_id"]),
                int(transaction["amount_cents"]),
                datetime.fromisoformat(str(transaction["occurred_at"])),
                str(transaction["description"]),
                str(transaction.get("currency", "USD")),
                str(transaction["provider_transaction_id"])
                if transaction.get("provider_transaction_id") is not None
                else None,
                str(transaction.get("status", "posted")),
                str(transaction.get("source_id", "fixture")),
            )
            for transaction in payload.get("transactions", [])
        )
        sources = tuple(
            ImportSource(
                source_id=str(source["source_id"]),
                source_type=str(source["source_type"]),
                content_hash=str(source["content_hash"]),
                imported_at=datetime.fromisoformat(str(source["imported_at"])),
                coverage_start=(
                    datetime.fromisoformat(str(source["coverage_start"]))
                    if source.get("coverage_start")
                    else None
                ),
                coverage_end=(
                    datetime.fromisoformat(str(source["coverage_end"]))
                    if source.get("coverage_end")
                    else None
                ),
                complete=bool(source.get("complete", True)),
            )
            for source in payload.get("sources", [])
        )
        return FinanceSnapshot(
            owner_id=owner_id,
            accounts=accounts,
            transactions=transactions,
            provider_id=str(row[1]),
            captured_at=cast(datetime | None, row[2]),
            sources=sources,
        )


class FinanceLedger:
    """Canonical private ledger; provider data enters through a typed snapshot port."""

    def __init__(self, store: FinanceSnapshotStore | None = None) -> None:
        self._snapshots: dict[str, FinanceSnapshot] = {}
        self.store = store

    def record_snapshot(self, snapshot: FinanceSnapshot) -> None:
        if snapshot.owner_id == "":
            raise ValueError("finance owner is required")
        if any(account.owner_id != snapshot.owner_id for account in snapshot.accounts):
            raise ValueError("account owner mismatch")
        account_ids = {account.account_id for account in snapshot.accounts}
        if len(account_ids) != len(snapshot.accounts):
            raise ValueError("duplicate account id")
        if any(transaction.account_id not in account_ids for transaction in snapshot.transactions):
            raise ValueError("transaction references unknown account")
        if self.store is None:
            self._snapshots[snapshot.owner_id] = snapshot
        else:
            self.store.save(snapshot)

    def import_csv(
        self,
        owner_id: str,
        account_id: str,
        content: str,
        *,
        source_id: str,
        currency: str = "USD",
        imported_at: datetime | None = None,
    ) -> FinanceImportReport:
        """Merge one owner-controlled import without replacing prior coverage."""

        snapshot = self._snapshot(owner_id)
        if account_id not in {account.account_id for account in snapshot.accounts}:
            raise KeyError("finance account is unavailable")
        transactions, report = import_csv_transactions(
            content,
            owner_id=owner_id,
            account_id=account_id,
            source_id=source_id,
            currency=currency,
            imported_at=imported_at,
        )
        if any(source.content_hash == report.source.content_hash for source in snapshot.sources):
            return report
        existing_ids = {item.transaction_id for item in snapshot.transactions}
        existing_provider_ids = {
            item.provider_transaction_id
            for item in snapshot.transactions
            if item.provider_transaction_id is not None
        }
        additions = tuple(
            item
            for item in transactions
            if item.transaction_id not in existing_ids
            and item.provider_transaction_id not in existing_provider_ids
        )
        merged = FinanceSnapshot(
            owner_id=snapshot.owner_id,
            accounts=snapshot.accounts,
            transactions=snapshot.transactions + additions,
            provider_id=snapshot.provider_id,
            captured_at=snapshot.captured_at,
            sources=snapshot.sources + (report.source,),
        )
        self.record_snapshot(merged)
        return report

    def _snapshot(self, owner_id: str) -> FinanceSnapshot:
        snapshot = (
            self._snapshots.get(owner_id) if self.store is None else self.store.load(owner_id)
        )
        if snapshot is None:
            raise KeyError("finance snapshot is unavailable")
        return snapshot

    def private_snapshot(self, requester: Principal, owner_id: str) -> FinanceSnapshot:
        if requester.id != owner_id:
            raise PermissionError("private finance state belongs to another principal")
        return self._snapshot(owner_id)

    def provenance(self, requester: Principal, owner_id: str) -> dict[str, str | None]:
        snapshot = self.private_snapshot(requester, owner_id)
        return {
            "provider_id": snapshot.provider_id,
            "captured_at": snapshot.captured_at.isoformat() if snapshot.captured_at else None,
        }

    def total_balance(self, requester: Principal, owner_id: str) -> int:
        return sum(
            account.balance_cents for account in self.private_snapshot(requester, owner_id).accounts
        )

    def derived_contribution(
        self,
        requester: Principal,
        owner_id: str,
        space_id: str,
        amount_cents: int,
        may_derive: bool,
    ) -> PrivateContribution:
        if not may_derive:
            raise PermissionError("finance contribution projection was not authorized")
        self.private_snapshot(requester, owner_id)
        if amount_cents < 0:
            raise ValueError("derived contribution cannot be negative")
        return PrivateContribution(owner_id, amount_cents, f"finance-contribution:{space_id}")

    def assess_affordability(
        self,
        requester: Principal,
        owner_id: str,
        purchase_cents: int,
        obligations: tuple[SharedObligation, ...],
        reserve_cents: int = 0,
    ) -> AffordabilityProjection:
        if purchase_cents < 0 or reserve_cents < 0:
            raise ValueError("purchase and reserve amounts cannot be negative")
        balance = self.total_balance(requester, owner_id)
        obligations_total = sum(obligation.amount for obligation in obligations)
        available = balance - obligations_total - reserve_cents
        return AffordabilityProjection(
            purchase_cents=purchase_cents,
            shared_obligations_cents=obligations_total,
            affordable=available >= purchase_cents,
            shortfall_cents=max(0, purchase_cents - available),
        )


class FinanceReadFastPath:
    """Deterministic affordability read using private state only below Core."""

    _AMOUNT = re.compile(r"(?:\$\s*|usd\s*)(\d+(?:\.\d{1,2})?)", re.IGNORECASE)
    _SPOKEN_AMOUNT = re.compile(
        r"\b(?P<amount>(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
        r"(?:(?:\s+|-)(?:one|two|three|four|five|six|seven|eight|nine))?)\s+"
        r"(?:dollars?|bucks?)\b",
        re.IGNORECASE,
    )
    _NUMBER_WORDS = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    _SAFE_PURCHASE = re.compile(
        r"\b(?:purchase|expense|spend(?:ing)?)\b.{0,40}\b(?:safe|okay|ok|manageable)\b",
        re.IGNORECASE,
    )
    _SAFE_SPEND = re.compile(
        r"\b(?:safe|okay|ok|manageable)\b.{0,40}\bspend(?:ing)?\b",
        re.IGNORECASE,
    )

    def __init__(self, ledger: FinanceLedger) -> None:
        self.ledger = ledger

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        if has_multiple_question_clauses(text):
            # One affordability Result cannot complete a compound objective;
            # bounded cognition must clarify or compose it instead.
            return False
        has_amount = cls.amount_cents(text) is not None
        return has_amount and (
            "can i afford" in text
            or "can we afford" in text
            or bool(cls._SAFE_PURCHASE.search(text))
            or bool(cls._SAFE_SPEND.search(text))
        )

    @classmethod
    def unsupported_balance_read(cls, utterance: str) -> bool:
        """Reject unsupported balance reads before cognition can misroute them."""
        text = utterance.casefold()
        return "balance" in text and not cls.matches(text) and not cls.needs_purchase_amount(text)

    @classmethod
    def needs_purchase_amount(cls, utterance: str) -> bool:
        """Identify affordability questions that cannot be answered without an amount."""

        text = utterance.casefold()
        return cls.amount_cents(text) is None and (
            "can i afford" in text
            or "can we afford" in text
            or "affordable" in text
            or "safe to spend" in text
        )

    @classmethod
    def amount_cents(cls, utterance: str) -> int | None:
        """Parse bounded numeric or spoken dollar amounts without model routing."""

        numeric = cls._AMOUNT.search(utterance)
        if numeric is not None:
            return round(float(numeric.group(1)) * 100)
        spoken = cls._SPOKEN_AMOUNT.search(utterance)
        if spoken is None:
            return None
        amount = sum(
            cls._NUMBER_WORDS[word]
            for word in spoken.group("amount").casefold().replace("-", " ").split()
        )
        return amount * 100

    def resolve(
        self,
        intent: IntentFrame,
        obligations: tuple[SharedObligation, ...] = (),
    ) -> Result | None:
        if not self.matches(intent.utterance):
            return None
        purchase_cents = self.amount_cents(intent.utterance)
        if purchase_cents is None:
            return None
        projection = self.ledger.assess_affordability(
            intent.principal,
            intent.principal.id,
            purchase_cents,
            obligations,
        )
        status = "can afford" if projection.affordable else "cannot afford"
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message=f"Based on your available snapshot, you {status} ${purchase_cents / 100:.2f}.",
            evidence={
                "affordable": projection.affordable,
                "purchase_cents": projection.purchase_cents,
                "shared_obligations_cents": projection.shared_obligations_cents,
                "shortfall_cents": projection.shortfall_cents,
            },
            correlation_id=intent.correlation_id,
        )
