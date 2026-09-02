"""Architectural guardrails for generic Pack/Core modules."""

from pathlib import Path

GENERIC_CORE = (
    Path("src/aegis/kernel.py"),
    Path("src/aegis/pack_lifecycle.py"),
    Path("src/aegis/pack_runtime.py"),
    Path("src/aegis/registry.py"),
)
PACK_ACTION_IDS = (
    "tasks.",
    "kitchen.",
    "finance.",
    "homelab.",
    "network.",
)
DOMAIN_IMPORTS = (
    "from .tasks",
    "from .household",
    "from .personal",
    "from .finance",
    "from .reference_packs",
)


def test_generic_core_does_not_embed_first_party_action_names() -> None:
    violations: list[str] = []
    for path in GENERIC_CORE:
        source = path.read_text(encoding="utf-8")
        for marker in PACK_ACTION_IDS:
            if marker in source:
                violations.append(f"{path}: {marker}")
    assert not violations, "domain-specific action IDs leaked into generic Core: " + ", ".join(
        violations
    )


def test_generic_core_does_not_import_first_party_domain_implementations() -> None:
    violations: list[str] = []
    for path in GENERIC_CORE:
        source = path.read_text(encoding="utf-8")
        for marker in DOMAIN_IMPORTS:
            if marker in source:
                violations.append(f"{path}: {marker}")
    assert not violations, "domain implementations leaked into generic Core: " + ", ".join(
        violations
    )


def test_pack_manager_exposes_metadata_without_private_state_access() -> None:
    source = Path("src/aegis/interaction.py").read_text(encoding="utf-8")
    assert "manager._bundles" not in source


def test_pack_lifecycle_snapshot_is_the_projection_contract() -> None:
    source = Path("src/aegis/cli.py").read_text(encoding="utf-8")
    assert ".lifecycle_snapshot()" in source
    assert "PostgresPackStore(connection).load()" not in source


def test_production_composition_supplies_pack_lifecycle_contract() -> None:
    source = Path("src/aegis/cli.py").read_text(encoding="utf-8")
    assert "pack_bundles=reference_bundles" in source
    assert "auto_enable_pack_ids" in source
