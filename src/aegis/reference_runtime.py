"""Composition-only runtime adapters for legacy first-party Pack actions."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .contracts import Principal
from .gateway_rpc import OpenClawWebSocketChannel
from .household import (
    PostgresChoreExecutor,
    PostgresChoreVerifier,
    PostgresEventExecutor,
    PostgresEventVerifier,
    PostgresHouseholdStore,
)
from .identity import Role
from .openclaw import OpenClawExecutor
from .pack_runtime import ActionRuntime, PackRuntimeRegistry
from .reference_packs import (
    OpenClawGroceryExecutor,
    OpenClawGroceryVerifier,
    OpenClawHomelabExecutor,
    OpenClawHomelabVerifier,
    OpenClawNetworkProbeExecutor,
    OpenClawNetworkProbeVerifier,
    PostgresGroceryListExecutor,
    PostgresGroceryListVerifier,
)
from .tasks import (
    PostgresTaskExecutor,
    PostgresTaskListExecutor,
    PostgresTaskListVerifier,
    PostgresTaskStore,
    PostgresTaskVerifier,
)


class _RuntimePolicy:
    def allows(self, request: Any) -> bool:
        return bool(request.action.action_id == "kitchen.groceries.add")


class _NoApproval:
    def required(self, request: Any) -> bool:
        return False

    def approved(self, request: Any) -> bool:
        return True


def default_runtime_registry(
    openclaw_channel: Callable[[], OpenClawWebSocketChannel],
) -> PackRuntimeRegistry:
    """Compose reference Pack runtimes without leaking action knowledge to clients."""

    registry = PackRuntimeRegistry()

    def task_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        store = PostgresTaskStore(connection)
        return ActionRuntime(
            PostgresTaskExecutor(store, principal),
            PostgresTaskVerifier(store, principal),
            {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def task_list_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        store = PostgresTaskStore(connection)
        return ActionRuntime(
            PostgresTaskListExecutor(store, principal),
            PostgresTaskListVerifier(store, principal),
            {"tasks.read": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def household_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        store = PostgresHouseholdStore(connection)
        return ActionRuntime(
            PostgresChoreExecutor(store, principal),
            PostgresChoreVerifier(store, principal),
            {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def event_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        store = PostgresHouseholdStore(connection)
        return ActionRuntime(
            PostgresEventExecutor(store, principal),
            PostgresEventVerifier(store, principal),
            {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def grocery_list_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        store = PostgresHouseholdStore(connection)
        return ActionRuntime(
            PostgresGroceryListExecutor(store, principal),
            PostgresGroceryListVerifier(store, principal),
            {"kitchen.read": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def grocery_add_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        channel = openclaw_channel()
        store = PostgresHouseholdStore(connection)
        return ActionRuntime(
            OpenClawExecutor(
                OpenClawGroceryExecutor(
                    channel,
                    os.environ.get("AEGIS_LIVE_GROCERY_PATH", "/tmp/aegis-alpha-groceries.tsv"),
                    store,
                    principal,
                ),
                _RuntimePolicy(),
                _NoApproval(),
            ),
            OpenClawGroceryVerifier(store, principal),
            {"kitchen.write": frozenset({Role.OWNER, Role.MEMBER})},
            cleanup=channel.close,
        )

    def homelab_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection, principal
        channel = openclaw_channel()
        services = {
            name.removeprefix("AEGIS_HOMELAB_SERVICE_").lower(): command
            for name, command in os.environ.items()
            if name.startswith("AEGIS_HOMELAB_SERVICE_") and command
        }
        endpoints = {
            name.removeprefix("AEGIS_HOMELAB_HEALTH_").lower(): endpoint
            for name, endpoint in os.environ.items()
            if name.startswith("AEGIS_HOMELAB_HEALTH_") and endpoint
        }
        return ActionRuntime(
            OpenClawExecutor(
                OpenClawHomelabExecutor(channel, services), _RuntimePolicy(), _NoApproval()
            ),
            OpenClawHomelabVerifier(endpoints),
            {"homelab.service.restart": frozenset({Role.OWNER})},
            cleanup=channel.close,
        )

    def network_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection, principal
        channel = openclaw_channel()
        return ActionRuntime(
            OpenClawExecutor(
                OpenClawNetworkProbeExecutor(channel), _RuntimePolicy(), _NoApproval()
            ),
            OpenClawNetworkProbeVerifier(),
            {"network.read": frozenset({Role.OWNER, Role.MEMBER})},
            cleanup=channel.close,
        )

    from .reference_packs import reference_bundles

    factories: dict[str, Callable[[Any, Principal], ActionRuntime]] = {
        "tasks.create": task_runtime,
        "tasks.complete": task_runtime,
        "tasks.list": task_list_runtime,
        "tasks.chores.create": household_runtime,
        "tasks.chores.complete": household_runtime,
        "tasks.events.create": event_runtime,
        "kitchen.groceries.list": grocery_list_runtime,
        "kitchen.groceries.add": grocery_add_runtime,
        "homelab.service.restart": homelab_runtime,
        "network.probe": network_runtime,
    }
    for bundle in reference_bundles():
        card_factories = {
            card.action.action_id: factories[card.action.action_id] for card in bundle.cards
        }
        registry.register_pack(bundle.cards, card_factories)
    return registry


def legacy_runtime(
    action_id: str,
    connection: Any,
    principal: Principal,
    openclaw_channel: Callable[[], OpenClawWebSocketChannel],
) -> ActionRuntime:
    """Resolve old first-party adapters outside the generic interaction service."""

    if action_id == "kitchen.groceries.add":
        channel = openclaw_channel()
        executor = OpenClawExecutor(
            OpenClawGroceryExecutor(
                channel,
                os.environ.get("AEGIS_LIVE_GROCERY_PATH", "/tmp/aegis-alpha-groceries.tsv"),
                PostgresHouseholdStore(connection),
                principal,
            ),
            _RuntimePolicy(),
            _NoApproval(),
        )
        return ActionRuntime(
            executor,
            OpenClawGroceryVerifier(PostgresHouseholdStore(connection), principal),
            {"kitchen.write": frozenset({Role.OWNER, Role.MEMBER})},
            cleanup=channel.close,
        )
    if action_id == "kitchen.groceries.list":
        store = PostgresHouseholdStore(connection)
        return ActionRuntime(
            PostgresGroceryListExecutor(store, principal),
            PostgresGroceryListVerifier(store, principal),
            {"kitchen.read": frozenset({Role.OWNER, Role.MEMBER})},
        )
    if action_id in {"tasks.create", "tasks.complete"}:
        task_store = PostgresTaskStore(connection)
        return ActionRuntime(
            PostgresTaskExecutor(task_store, principal),
            PostgresTaskVerifier(task_store, principal),
            {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})},
        )
    if action_id == "tasks.list":
        task_store = PostgresTaskStore(connection)
        return ActionRuntime(
            PostgresTaskListExecutor(task_store, principal),
            PostgresTaskListVerifier(task_store, principal),
            {"tasks.read": frozenset({Role.OWNER, Role.MEMBER})},
        )
    if action_id in {"tasks.chores.create", "tasks.chores.complete"}:
        store = PostgresHouseholdStore(connection)
        return ActionRuntime(
            PostgresChoreExecutor(store, principal),
            PostgresChoreVerifier(store, principal),
            {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})},
        )
    if action_id == "tasks.events.create":
        store = PostgresHouseholdStore(connection)
        return ActionRuntime(
            PostgresEventExecutor(store, principal),
            PostgresEventVerifier(store, principal),
            {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})},
        )
    if action_id == "homelab.service.restart":
        channel = openclaw_channel()
        services = {
            name.removeprefix("AEGIS_HOMELAB_SERVICE_").lower(): command
            for name, command in os.environ.items()
            if name.startswith("AEGIS_HOMELAB_SERVICE_") and command
        }
        endpoints = {
            name.removeprefix("AEGIS_HOMELAB_HEALTH_").lower(): endpoint
            for name, endpoint in os.environ.items()
            if name.startswith("AEGIS_HOMELAB_HEALTH_") and endpoint
        }
        return ActionRuntime(
            OpenClawExecutor(
                OpenClawHomelabExecutor(channel, services), _RuntimePolicy(), _NoApproval()
            ),
            OpenClawHomelabVerifier(endpoints),
            {"homelab.service.restart": frozenset({Role.OWNER})},
            cleanup=channel.close,
        )
    if action_id == "network.probe":
        channel = openclaw_channel()
        return ActionRuntime(
            OpenClawExecutor(
                OpenClawNetworkProbeExecutor(channel), _RuntimePolicy(), _NoApproval()
            ),
            OpenClawNetworkProbeVerifier(),
            {"network.read": frozenset({Role.OWNER, Role.MEMBER})},
            cleanup=channel.close,
        )
    raise LookupError(f"no legacy runtime binding for {action_id}")
