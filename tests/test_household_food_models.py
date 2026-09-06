from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from aegis.contracts import ActionSpec, ExecutionRequest, VerificationContract
from aegis.household import (
    HouseholdSpace,
    PantryItem,
    migrate_grocery_strings,
    normalize_food_key,
)
from aegis.reference_packs import (
    PostgresGroceryStateExecutor,
    PostgresGroceryStateVerifier,
    PostgresPantryMutationExecutor,
    PostgresPantryMutationVerifier,
)


def test_legacy_groceries_migrate_to_stable_ids_without_inventing_facts():
    first = migrate_grocery_strings(["milk", " eggs "])
    second = migrate_grocery_strings(["milk", " eggs "])

    assert first == second
    assert [item.display_name for item in first.values()] == ["milk", "eggs"]
    assert all(item.desired_quantity is None for item in first.values())
    assert all(item.unit is None for item in first.values())
    assert all(item.state == "needed" for item in first.values())
    assert all(item.pantry_item_id is None for item in first.values())


def test_household_space_exposes_structured_food_without_breaking_legacy_projection():
    space = HouseholdSpace("kitchen", {"owner"}, groceries=["Milk"])

    principal = SimpleNamespace(id="owner", space_ids=("kitchen",))
    assert tuple(space.groceries) == ("Milk",)
    assert len(space.grocery_items) == 1
    item = next(iter(space.grocery_items.values()))
    assert item.normalized_key == "milk"
    assert space.snapshot(principal)["grocery_items"] == (item,)

    space.add_grocery(principal, "Bread", "add-bread-1")
    assert [item.display_name for item in space.grocery_items.values()] == ["Milk", "Bread"]
    assert normalize_food_key("  BREAD  ") == "bread"


def test_food_mutations_use_stable_ids_and_reject_stale_pantry_writes():
    principal = SimpleNamespace(id="owner", space_ids=("kitchen",))
    space = HouseholdSpace("kitchen", {"owner"}, groceries=["Milk"])
    grocery_id = next(iter(space.grocery_items))

    purchased = space.mark_grocery_purchased(principal, grocery_id)
    assert purchased.state == "purchased"
    assert purchased.version == 1

    pantry = PantryItem("pantry-milk", "Milk", "milk", quantity=2, unit="carton")
    space.add_pantry(principal, pantry)
    consumed = space.consume_pantry(principal, pantry.item_id, 1, expected_version=0)
    assert consumed.quantity == 1
    assert consumed.version == 1
    with pytest.raises(ValueError, match="stale pantry"):
        space.consume_pantry(principal, pantry.item_id, 1, expected_version=0)


def test_unknown_pantry_quantity_is_not_treated_as_zero():
    principal = SimpleNamespace(id="owner", space_ids=("kitchen",))
    space = HouseholdSpace("kitchen", {"owner"})
    item = PantryItem("pantry-unknown", "Rice", "rice")
    space.add_pantry(principal, item)
    with pytest.raises(ValueError, match="quantity is unknown"):
        space.consume_pantry(principal, item.item_id, 1, expected_version=0)


def test_store_load_migrates_legacy_payload_once():
    from aegis.household import PostgresHouseholdStore

    class Connection:
        def execute(self, query, params=()):
            del params
            assert "SELECT payload" in query
            return self

        def fetchone(self):
            return ({"groceries": ["milk"]},)

    space = PostgresHouseholdStore(Connection()).load("kitchen", {"owner"})
    item = next(iter(space.grocery_items.values()))
    assert item.display_name == "milk"
    assert item.desired_quantity is None


def test_kitchen_pack_declares_pantry_read_as_a_generic_capability():
    from aegis.reference_packs import reference_bundles

    kitchen = next(bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "kitchen")
    card = next(card for card in kitchen.cards if card.action.action_id == "kitchen.pantry.list")
    assert card.action.required_permissions == ("kitchen.read",)
    assert card.semantic_scope == "kitchen.pantry"


def test_pantry_pack_mutations_preserve_unknowns_and_verify_scoped_state():
    principal = SimpleNamespace(id="owner", space_ids=("kitchen",))
    space = HouseholdSpace("kitchen", {"owner"})

    class Store:
        def add_pantry_item(self, _principal, item):
            return space.add_pantry(principal, item)

        def update_pantry_item(self, _principal, item, expected_version):
            return space.update_pantry(principal, item, expected_version)

        def consume_pantry_item(self, _principal, item_id, quantity, expected_version):
            return space.consume_pantry(principal, item_id, quantity, expected_version)

        def list_pantry_items(self, _principal):
            return tuple(space.pantry_items.values())

    store = Store()
    add = ActionSpec(
        action_id="kitchen.pantry.add",
        capability="kitchen.pantry.write",
        arguments={"item_id": "pantry-milk", "display_name": "Milk"},
    )
    request = ExecutionRequest(
        objective_id=uuid4(), action_id=uuid4(), action=add, idempotency_key="pantry-add-1"
    )
    observation = PostgresPantryMutationExecutor(store, principal).execute(request)
    result = PostgresPantryMutationVerifier(store, principal).verify(
        observation, VerificationContract(kind="readback")
    )
    assert result.verified
    assert space.pantry_items["pantry-milk"].quantity is None

    consume = add.model_copy(
        update={
            "action_id": "kitchen.pantry.consume",
            "arguments": {"item_id": "pantry-milk", "quantity": 1, "expected_version": 0},
        }
    )
    failed = PostgresPantryMutationExecutor(store, principal).execute(
        ExecutionRequest(
            objective_id=uuid4(), action_id=uuid4(), action=consume, idempotency_key="consume-1"
        )
    )
    assert not failed.command_succeeded


def test_grocery_state_action_uses_id_and_independent_current_read():
    principal = SimpleNamespace(id="owner", space_ids=("kitchen",))
    space = HouseholdSpace("kitchen", {"owner"}, groceries=["Milk"])
    grocery_id = next(iter(space.grocery_items))

    class Store:
        def mark_grocery_purchased(self, _principal, item_id):
            return space.mark_grocery_purchased(principal, item_id)

        def remove_grocery(self, _principal, item_id):
            return space.remove_grocery(principal, item_id)

        def list_grocery_items(self, _principal):
            return tuple(space.grocery_items.values())

    action = ActionSpec(
        action_id="kitchen.groceries.mark_purchased",
        capability="kitchen.groceries.write",
        arguments={"grocery_id": grocery_id},
    )
    request = ExecutionRequest(
        objective_id=uuid4(), action_id=uuid4(), action=action, idempotency_key="purchase-1"
    )
    observation = PostgresGroceryStateExecutor(Store(), principal).execute(request)
    assert (
        PostgresGroceryStateVerifier(Store(), principal)
        .verify(observation, VerificationContract(kind="readback"))
        .verified
    )
