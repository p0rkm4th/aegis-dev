"""Pack manifest validation and lifecycle; installation never grants authority implicitly."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditLog
from .contracts import ActionCard
from .registry import CapabilityEmbedder, CapabilityRegistry


class PackStatus(StrEnum):
    DISCOVERED = "discovered"
    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"


class PackUI(BaseModel):
    """Optional presentation hints; these fields never grant Pack authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=1)
    category: str = Field(min_length=1)
    detail_view: str | None = None


class PackManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pack_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9-]*$")
    version: str = Field(min_length=1)
    permissions: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    ui: PackUI | None = None


class PackBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    manifest: PackManifest
    cards: tuple[ActionCard, ...] = Field(max_length=100)


class PackStore(Protocol):
    def save(
        self,
        bundle: PackBundle,
        status: PackStatus,
        granted_permissions: frozenset[str],
    ) -> None: ...

    def load(self) -> tuple[tuple[PackBundle, PackStatus, frozenset[str]], ...]: ...

    def delete(self, pack_id: str) -> None: ...


class PostgresPackStore:
    """Persist Pack lifecycle state without making PostgreSQL semantic owner."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def save(
        self,
        bundle: PackBundle,
        status: PackStatus,
        granted_permissions: frozenset[str],
    ) -> None:
        self.connection.execute(
            """INSERT INTO pack_installations
               (pack_id, version, manifest, cards, granted_permissions, status)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (pack_id) DO UPDATE SET version = EXCLUDED.version,
                 manifest = EXCLUDED.manifest, cards = EXCLUDED.cards,
                 granted_permissions = EXCLUDED.granted_permissions,
                 status = EXCLUDED.status, updated_at = now()""",
            (
                bundle.manifest.pack_id,
                bundle.manifest.version,
                json.dumps(bundle.manifest.model_dump(mode="json"), sort_keys=True),
                json.dumps([card.model_dump(mode="json") for card in bundle.cards], sort_keys=True),
                json.dumps(sorted(granted_permissions)),
                status.value,
            ),
        )
        self.connection.commit()

    def load(self) -> tuple[tuple[PackBundle, PackStatus, frozenset[str]], ...]:
        rows = self.connection.execute(
            "SELECT manifest, cards, granted_permissions, status "
            "FROM pack_installations ORDER BY pack_id"
        ).fetchall()
        loaded: list[tuple[PackBundle, PackStatus, frozenset[str]]] = []
        for manifest, cards, grants, status in rows:
            manifest_data = manifest if isinstance(manifest, dict) else json.loads(str(manifest))
            cards_data = cards if isinstance(cards, list) else json.loads(str(cards))
            grants_data = grants if isinstance(grants, list) else json.loads(str(grants))
            loaded.append(
                (
                    PackBundle.model_validate({"manifest": manifest_data, "cards": cards_data}),
                    PackStatus(str(status)),
                    frozenset(str(permission) for permission in grants_data),
                )
            )
        return tuple(loaded)

    def delete(self, pack_id: str) -> None:
        self.connection.execute("DELETE FROM pack_installations WHERE pack_id = %s", (pack_id,))
        self.connection.commit()


class PackManager:
    def __init__(self, audit: AuditLog | None = None, store: PackStore | None = None) -> None:
        self._bundles: dict[str, PackBundle] = {}
        self._statuses: dict[str, PackStatus] = {}
        self._grants: dict[str, frozenset[str]] = {}
        self.audit = audit or AuditLog()
        self.store = store
        if self.store is not None:
            for bundle, status, grants in self.store.load():
                self._bundles[bundle.manifest.pack_id] = bundle
                self._statuses[bundle.manifest.pack_id] = status
                self._grants[bundle.manifest.pack_id] = grants

    def discover(self, bundle: PackBundle, actor_id: str = "system") -> None:
        self._validate(bundle)
        if bundle.manifest.pack_id in self._bundles:
            raise ValueError("Pack is already discovered")
        self._bundles[bundle.manifest.pack_id] = bundle
        self._statuses[bundle.manifest.pack_id] = PackStatus.DISCOVERED
        self._persist(bundle)
        self.audit.append(
            "pack.discovered",
            actor_id,
            {"pack_id": bundle.manifest.pack_id, "version": bundle.manifest.version},
        )

    def install(
        self,
        pack_id: str,
        granted_permissions: frozenset[str] = frozenset(),
        actor_id: str = "system",
    ) -> None:
        bundle = self._require(pack_id)
        required = frozenset(bundle.manifest.permissions)
        if not required.issubset(granted_permissions):
            missing = sorted(required - granted_permissions)
            raise PermissionError(f"Pack permissions not granted: {missing}")
        self._grants[pack_id] = granted_permissions & required
        self._statuses[pack_id] = PackStatus.INSTALLED
        self._persist(bundle)
        self.audit.append(
            "pack.installed",
            actor_id,
            {"pack_id": pack_id, "permissions": sorted(self._grants[pack_id])},
        )

    def enable(self, pack_id: str, actor_id: str = "system") -> None:
        self._require_status(pack_id, PackStatus.INSTALLED, PackStatus.DISABLED)
        self._statuses[pack_id] = PackStatus.ENABLED
        self._persist(self._bundles[pack_id])
        self.audit.append("pack.enabled", actor_id, {"pack_id": pack_id})

    def disable(self, pack_id: str, actor_id: str = "system") -> None:
        self._require_status(pack_id, PackStatus.ENABLED)
        self._statuses[pack_id] = PackStatus.DISABLED
        self._persist(self._bundles[pack_id])
        self.audit.append("pack.disabled", actor_id, {"pack_id": pack_id})

    def remove(self, pack_id: str, actor_id: str = "system") -> None:
        self._require(pack_id)
        del self._bundles[pack_id]
        self._statuses.pop(pack_id, None)
        self._grants.pop(pack_id, None)
        if self.store is not None:
            self.store.delete(pack_id)
        self.audit.append("pack.removed", actor_id, {"pack_id": pack_id})

    def status(self, pack_id: str) -> PackStatus:
        self._require(pack_id)
        return self._statuses[pack_id]

    def enabled_cards(self) -> tuple[ActionCard, ...]:
        return tuple(
            card
            for pack_id, bundle in self._bundles.items()
            if self._statuses[pack_id] is PackStatus.ENABLED
            for card in bundle.cards
        )

    def retrieve(self, domain: str, limit: int = 5) -> tuple[ActionCard, ...]:
        """Retrieve only enabled Pack capabilities through Core's bounded registry."""
        return CapabilityRegistry(self.enabled_cards()).retrieve(domain, limit)

    def retrieve_semantic(
        self, query: str, embedder: CapabilityEmbedder, limit: int = 5
    ) -> tuple[ActionCard, ...]:
        """Retrieve a bounded semantic shortlist from enabled Pack metadata."""
        return CapabilityRegistry(self.enabled_cards()).retrieve_semantic(query, embedder, limit)

    def action_card(self, domain: str, action_id: str) -> ActionCard | None:
        """Find one exact enabled card without widening the model shortlist."""
        return next(
            (
                card
                for card in self.enabled_cards()
                if domain in card.action.capability and card.action.action_id == action_id
            ),
            None,
        )

    def granted_permissions(self, pack_id: str) -> frozenset[str]:
        self._require(pack_id)
        return self._grants.get(pack_id, frozenset())

    def _require(self, pack_id: str) -> PackBundle:
        try:
            return self._bundles[pack_id]
        except KeyError as exc:
            raise KeyError(f"unknown Pack: {pack_id}") from exc

    def _require_status(self, pack_id: str, *allowed: PackStatus) -> None:
        self._require(pack_id)
        if self._statuses[pack_id] not in allowed:
            raise ValueError(f"invalid lifecycle transition from {self._statuses[pack_id]}")

    def _persist(self, bundle: PackBundle) -> None:
        if self.store is not None:
            self.store.save(
                bundle,
                self._statuses[bundle.manifest.pack_id],
                self._grants.get(bundle.manifest.pack_id, frozenset()),
            )

    @staticmethod
    def _validate(bundle: PackBundle) -> None:
        ids = [card.action.action_id for card in bundle.cards]
        if len(ids) != len(set(ids)):
            raise ValueError("Pack contains duplicate action ids")
        declared = set(bundle.manifest.permissions)
        for card in bundle.cards:
            prefix = f"{bundle.manifest.pack_id}."
            if not card.action.action_id.startswith(prefix):
                raise ValueError("Pack action id must remain within its Pack namespace")
            if not card.action.capability.startswith(prefix):
                raise ValueError("Pack capability must remain within its Pack namespace")
            if not set(card.action.required_permissions).issubset(declared):
                raise ValueError("Action requires an undeclared Pack permission")
            if card.action.required_permissions and card.action.verification is None:
                raise ValueError("permissioned Pack actions require verification")
