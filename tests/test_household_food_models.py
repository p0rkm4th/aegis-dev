from __future__ import annotations

from types import SimpleNamespace

from aegis.household import HouseholdSpace, migrate_grocery_strings, normalize_food_key


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
