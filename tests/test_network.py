from __future__ import annotations

import pytest

from aegis.homelab import Host, classify_discovered_device
from aegis.network import (
    AuthorizedNetworkScope,
    BoundedNetworkDiscovery,
    DiscoveredDevice,
    HomelabInventory,
    ScopeDenied,
)


def test_discovery_reconciliation_is_candidate_only_when_stable_evidence_matches() -> None:
    host = Host(
        "atlas",
        "192.0.2.10",
        "atlas.local",
        known_addresses=("192.0.2.11",),
        identity_evidence=("owner-configured hostname",),
    )
    candidate = classify_discovered_device(
        DiscoveredDevice("192.0.2.11", hostname="atlas.local"), {host.host_id: host}
    )
    assert candidate == {
        "identity_status": "reconciliation_candidate",
        "canonical_host_id": "atlas",
    }


def test_address_only_discovery_never_becomes_canonical_identity() -> None:
    host = Host("atlas", "192.0.2.10", "atlas.local", identity_evidence=("fixture",))
    result = classify_discovered_device(DiscoveredDevice("192.0.2.10"), {host.host_id: host})
    assert result == {
        "identity_status": "unmatched_observation",
        "canonical_host_id": None,
    }


def test_bounded_discovery_returns_observations_without_promoting_hosts() -> None:
    inventory = HomelabInventory(
        scopes={"lab": AuthorizedNetworkScope("lab", ("192.0.2.0/29",), "owner lab")}
    )
    seen: list[tuple[str, int]] = []

    def probe(address: str, port: int) -> bool:
        seen.append((address, port))
        return address == "192.0.2.2" and port == 443

    found = BoundedNetworkDiscovery(probe, max_hosts=4, ports=(22, 443)).discover(inventory, "lab")

    assert found == (DiscoveredDevice("192.0.2.2", services=("443",)),)
    assert len(seen) == 8
    assert inventory.devices == {}


def test_discovery_caps_hosts_and_rejects_unknown_or_inactive_scope() -> None:
    inventory = HomelabInventory(
        scopes={
            "lab": AuthorizedNetworkScope("lab", ("192.0.2.0/24",), "owner lab"),
            "off": AuthorizedNetworkScope("off", ("192.0.2.0/29",), "retired", False),
        }
    )
    seen: list[str] = []
    discovery = BoundedNetworkDiscovery(
        lambda address, _port: seen.append(address) or False,
        max_hosts=3,
        ports=(443,),
    )

    assert discovery.discover(inventory, "lab") == ()
    assert len(seen) == 3
    with pytest.raises(ScopeDenied, match="unknown authorization scope"):
        discovery.discover(inventory, "missing")
    with pytest.raises(ScopeDenied, match="inactive authorization scope"):
        discovery.discover(inventory, "off")


def test_discovery_configuration_is_bounded() -> None:
    with pytest.raises(ValueError, match="max_hosts"):
        BoundedNetworkDiscovery(lambda _address, _port: False, max_hosts=0)
    with pytest.raises(ValueError, match="valid TCP ports"):
        BoundedNetworkDiscovery(lambda _address, _port: False, ports=(0,))


def test_loopback_scope_does_not_turn_aliases_into_devices() -> None:
    inventory = HomelabInventory(
        scopes={"local": AuthorizedNetworkScope("local", ("127.0.0.0/8",), "owner local")}
    )
    seen: list[tuple[str, int]] = []

    def probe(address: str, port: int) -> bool:
        seen.append((address, port))
        return True

    found = BoundedNetworkDiscovery(probe, ports=(22, 443)).discover(inventory, "local")

    assert found == (DiscoveredDevice("127.0.0.1", services=("22", "443")),)
    assert seen == [("127.0.0.1", 22), ("127.0.0.1", 443)]
