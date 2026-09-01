"""Provider-independent private finance state and controlled derived outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from .contracts import IntentFrame, ObjectiveState, Principal, Result
from .projections import PrivateContribution, SharedObligation


@dataclass(frozen=True)
class Account:
    account_id: str
    owner_id: str
    balance_cents: int


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    account_id: str
    amount_cents: int
    occurred_at: datetime
    description: str


@dataclass(frozen=True)
class FinanceSnapshot:
    owner_id: str
    accounts: tuple[Account, ...]
    transactions: tuple[Transaction, ...] = ()
    provider_id: str = "fixture"
    captured_at: datetime | None = None


@dataclass(frozen=True)
class AffordabilityProjection:
    purchase_cents: int
    shared_obligations_cents: int
    affordable: bool
    shortfall_cents: int


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
            "SELECT payload, provider_id, captured_at FROM finance_snapshots "
            "WHERE owner_id = %s",
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
            )
            for transaction in payload.get("transactions", [])
        )
        return FinanceSnapshot(
            owner_id=owner_id,
            accounts=accounts,
            transactions=transactions,
            provider_id=str(row[1]),
            captured_at=cast(datetime | None, row[2]),
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

    def _snapshot(self, owner_id: str) -> FinanceSnapshot:
        snapshot = (
            self._snapshots.get(owner_id)
            if self.store is None
            else self.store.load(owner_id)
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

    def __init__(self, ledger: FinanceLedger) -> None:
        self.ledger = ledger

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        return ("can i afford" in text or "can we afford" in text) and bool(
            cls._AMOUNT.search(text)
        )

    def resolve(
        self,
        intent: IntentFrame,
        obligations: tuple[SharedObligation, ...] = (),
    ) -> Result | None:
        if not self.matches(intent.utterance):
            return None
        match = self._AMOUNT.search(intent.utterance)
        if match is None:
            return None
        purchase_cents = round(float(match.group(1)) * 100)
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
