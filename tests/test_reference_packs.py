from aegis.pack_lifecycle import PackBundle
from aegis.reference_packs import reference_bundles, reference_packs


def test_first_party_packs_use_the_generic_pack_bundle_contract() -> None:
    packs = reference_packs()

    assert packs
    assert all(isinstance(pack, PackBundle) for pack in packs)
    assert reference_bundles() == packs
    assert {pack.manifest.pack_id for pack in packs} == {
        "tasks",
        "kitchen",
        "homelab",
        "network",
    }
