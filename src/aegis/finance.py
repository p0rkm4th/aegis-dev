"""Provider-independent private finance state and controlled derived outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .contracts import Principal
from .projections import PrivateContribution


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


class FinancialProvider(Protocol):
    def snapshot(self, owner_id: str) -> FinanceSnapshot: ...


class FinanceLedger:
    """Canonical private ledger; provider data enters through a typed snapshot port."""

    def __init__(self) -> None:
        self._snapshots: dict[str, FinanceSnapshot] = {}

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
        self._snapshots[snapshot.owner_id] = snapshot

    def private_snapshot(self, requester: Principal, owner_id: str) -> FinanceSnapshot:
        if requester.id != owner_id:
            raise PermissionError("private finance state belongs to another principal")
        try:
            return self._snapshots[owner_id]
        except KeyError as exc:
            raise KeyError("finance snapshot is unavailable") from exc

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
