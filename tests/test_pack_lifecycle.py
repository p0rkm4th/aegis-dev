from __future__ import annotations

import pytest

from aegis.audit import AuditLog
from aegis.contracts import (
    ActionCard,
    ActionSpec,
    ArgumentGroundingRule,
    ArgumentProvenanceKind,
    VerificationContract,
)
from aegis.pack_lifecycle import PackBundle, PackManager, PackManifest, validate_pack_bundle


def bundle(version: str, permissions: tuple[str, ...]) -> PackBundle:
    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=f"driver.{permission.replace('.', '-')}",
                capability=f"driver.{permission}",
                required_permissions=(permission,),
                verification=VerificationContract(kind="readback"),
            ),
            summary=f"Use {permission}",
            relevance=1,
        )
        for permission in permissions
    )
    return PackBundle(
        manifest=PackManifest(pack_id="driver", version=version, permissions=permissions),
        cards=cards,
    )


def installed_manager() -> PackManager:
    manager = PackManager(audit=AuditLog())
    manager.discover(bundle("1", ("driver.read",)))
    manager.install("driver", frozenset({"driver.read"}))
    manager.enable("driver")
    return manager


def test_pack_grounding_rules_must_match_declared_arguments():
    card = ActionCard(
        action=ActionSpec(action_id="driver.write", capability="driver.write"),
        summary="Write a driver value",
        relevance=1,
        argument_grounding={
            "value": ArgumentGroundingRule(
                permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
            )
        },
    )
    candidate = PackBundle(
        manifest=PackManifest(pack_id="driver", version="1", permissions=()), cards=(card,)
    )

    with pytest.raises(ValueError, match="declared ActionCard arguments"):
        validate_pack_bundle(candidate)


def test_pack_grounding_rules_require_declared_canonical_and_default_contracts():
    canonical = ActionCard(
        action=ActionSpec(action_id="driver.read", capability="driver.read"),
        summary="Read a driver value",
        relevance=1,
        argument_keys=("value",),
        argument_grounding={
            "value": ArgumentGroundingRule(
                permitted_provenance=(ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT,)
            )
        },
    )
    candidate = PackBundle(
        manifest=PackManifest(pack_id="driver", version="1", permissions=()),
        cards=(canonical,),
    )

    with pytest.raises(ValueError, match="canonical source"):
        validate_pack_bundle(candidate)


def test_upgrade_permission_expansion_keeps_active_pack_and_requires_approval():
    manager = installed_manager()
    candidate = bundle("2", ("driver.read", "driver.write"))

    manager.reconcile((candidate,), auto_enable=frozenset({"driver"}))

    assert manager.bundle("driver").manifest.version == "1"
    assert manager.status("driver").value == "enabled"
    assert manager.granted_permissions("driver") == frozenset({"driver.read"})
    assert manager.pending_upgrade("driver") is not None
    assert {card.action.action_id for card in manager.enabled_cards()} == {"driver.driver-read"}

    with pytest.raises(PermissionError):
        manager.approve_upgrade("driver", frozenset({"driver.read"}), "alice")

    manager.approve_upgrade("driver", frozenset({"driver.read", "driver.write"}), "alice")
    assert manager.bundle("driver").manifest.version == "2"
    assert manager.granted_permissions("driver") == frozenset({"driver.read", "driver.write"})
    assert manager.pending_upgrade("driver") is None
    assert manager.audit.events[-1].event_type == "pack.upgrade.approved"
    assert manager.audit.events[-1].principal_id == "alice"


def test_upgrade_contraction_applies_and_rename_is_expansion():
    manager = PackManager(audit=AuditLog())
    initial = bundle("2", ("driver.read", "driver.write"))
    manager.discover(initial)
    manager.install("driver", frozenset({"driver.read", "driver.write"}))
    manager.enable("driver")

    manager.reconcile((bundle("3", ("driver.read",)),), auto_enable=frozenset({"driver"}))
    assert manager.bundle("driver").manifest.version == "3"
    assert manager.granted_permissions("driver") == frozenset({"driver.read"})
    assert manager.pending_upgrade("driver") is None

    renamed = bundle("4", ("driver.new-read",))
    manager.reconcile((renamed,), auto_enable=frozenset({"driver"}))
    assert manager.bundle("driver").manifest.version == "3"
    assert manager.pending_upgrade("driver") is not None
    assert manager.pending_upgrade("driver").requested_permissions == frozenset({"driver.new-read"})


def test_invalid_upgrade_does_not_replace_last_working_pack():
    manager = installed_manager()
    invalid = PackBundle(
        manifest=PackManifest(pack_id="driver", version="bad", permissions=()),
        cards=(
            ActionCard(
                action=ActionSpec(action_id="other.read", capability="driver.read"),
                summary="Invalid namespace",
                relevance=1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="action id"):
        manager.reconcile((invalid,))
    assert manager.bundle("driver").manifest.version == "1"
    assert manager.pending_upgrade("driver") is None
