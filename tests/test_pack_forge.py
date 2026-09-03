import importlib.util
from pathlib import Path

import pytest

from aegis.contracts import ActionCard, ActionSpec, VerificationContract
from aegis.pack_forge import PackProposalV0, compile_pack_proposal, materialize_pack_skeleton
from aegis.pack_lifecycle import PackManager


def aquarium_proposal() -> PackProposalV0:
    return PackProposalV0(
        pack_id="aquarium",
        version="0.1.0",
        permissions=("aquarium.read", "aquarium.write"),
        ui={"label": "Aquarium", "category": "demo", "detail_view": "list"},
        cards=(
            ActionCard(
                action=ActionSpec(
                    action_id="aquarium.water-test.read",
                    capability="aquarium.water-test.read",
                    required_permissions=("aquarium.read",),
                    verification=VerificationContract(kind="readback"),
                ),
                summary="Read a water test",
                relevance=1,
            ),
            ActionCard(
                action=ActionSpec(
                    action_id="aquarium.water-test.record",
                    capability="aquarium.water-test.record",
                    required_permissions=("aquarium.write",),
                    verification=VerificationContract(kind="readback"),
                ),
                summary="Record a water test",
                relevance=1,
                argument_keys=("ph",),
                argument_descriptions={"ph": "measured pH"},
            ),
        ),
    )


def test_proposal_compiles_deterministically_and_reuses_production_validator() -> None:
    proposal = aquarium_proposal()
    first = compile_pack_proposal(proposal)
    second = compile_pack_proposal(proposal)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")

    manager = PackManager()
    manager.discover(first)
    assert manager.bundle("aquarium") == first


def test_proposal_rejects_unknown_fields_and_malformed_namespace() -> None:
    with pytest.raises(ValueError):
        PackProposalV0.model_validate({**aquarium_proposal().model_dump(), "future": True})
    invalid = aquarium_proposal().model_copy(
        update={
            "cards": (
                aquarium_proposal()
                .cards[0]
                .model_copy(
                    update={
                        "action": aquarium_proposal()
                        .cards[0]
                        .action.model_copy(
                            update={"action_id": "other.read", "capability": "other.read"}
                        )
                    }
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="namespace"):
        compile_pack_proposal(invalid)


def test_shared_validator_rejects_undeclared_permission_and_missing_verification() -> None:
    with pytest.raises(ValueError, match="undeclared"):
        compile_pack_proposal(
            aquarium_proposal().model_copy(
                update={
                    "cards": (
                        aquarium_proposal()
                        .cards[0]
                        .model_copy(
                            update={
                                "action": aquarium_proposal()
                                .cards[0]
                                .action.model_copy(
                                    update={"required_permissions": ("aquarium.missing",)}
                                )
                            }
                        ),
                    )
                }
            )
        )
    with pytest.raises(ValueError, match="verification"):
        compile_pack_proposal(
            aquarium_proposal().model_copy(
                update={
                    "cards": (
                        aquarium_proposal()
                        .cards[0]
                        .model_copy(
                            update={
                                "action": aquarium_proposal()
                                .cards[0]
                                .action.model_copy(
                                    update={
                                        "required_permissions": ("aquarium.read",),
                                        "verification": None,
                                    }
                                )
                            }
                        ),
                    )
                }
            )
        )


def test_local_materializer_preview_and_output_are_structural_only(tmp_path: Path) -> None:
    proposal = aquarium_proposal()
    preview = materialize_pack_skeleton(proposal, tmp_path / "preview", preview=True)
    assert "README.md" in preview
    assert "runtime.py" in preview
    assert not (tmp_path / "preview").exists()

    output = tmp_path / "aquarium"
    files = materialize_pack_skeleton(proposal, output)
    assert "pack_manifest.json" in files
    assert "runtime.py" in files
    assert "tests/test_pack_contract.py" in files
    assert "provenance/OSS.md" in files
    assert "NotImplementedError" in (output / "runtime.py").read_text()
    assert "command_succeeded=True" not in (output / "runtime.py").read_text()


def test_generated_action_ids_can_be_compared_to_registry_bindings() -> None:
    proposal = aquarium_proposal()
    declared = {card.action.action_id for card in proposal.cards}
    generated = {card.action.action_id for card in compile_pack_proposal(proposal).cards}
    assert generated == declared


def test_generated_upgrade_uses_existing_lifecycle_authority() -> None:
    manager = PackManager()
    initial = aquarium_proposal().model_copy(
        update={
            "version": "0.1.0",
            "permissions": ("aquarium.read",),
            "cards": (aquarium_proposal().cards[0],),
        }
    )
    manager.discover(compile_pack_proposal(initial))
    manager.install("aquarium", frozenset({"aquarium.read"}))
    manager.enable("aquarium")

    manager.reconcile(
        (compile_pack_proposal(aquarium_proposal()),), auto_enable=frozenset({"aquarium"})
    )
    assert manager.bundle("aquarium").manifest.version == "0.1.0"
    assert manager.pending_upgrade("aquarium") is not None
    assert manager.granted_permissions("aquarium") == frozenset({"aquarium.read"})
    manager.approve_upgrade("aquarium", frozenset({"aquarium.read", "aquarium.write"}), "owner")
    assert manager.granted_permissions("aquarium") == frozenset({"aquarium.read", "aquarium.write"})


def test_materialized_runtime_factory_is_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "aquarium"
    materialize_pack_skeleton(aquarium_proposal(), output)
    spec = importlib.util.spec_from_file_location("generated_runtime", output / "runtime.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(NotImplementedError):
        module.runtime_factory(None, None)
