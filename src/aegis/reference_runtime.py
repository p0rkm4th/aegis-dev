"""Composition-only runtime adapters for legacy first-party Pack actions."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .calendar import configured_calendar_write_provider
from .communications import FixtureCommunicationSendProvider, OpenClawCliCommunicationSendProvider
from .contracts import Principal
from .devices import FixtureDeviceGateway, HomeAssistantRestControlGateway
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
    CalendarCommunicationDraftExecutor,
    CalendarCommunicationDraftVerifier,
    CalendarCreateExecutor,
    CalendarCreateVerifier,
    CalendarEventsExecutor,
    CalendarEventsVerifier,
    CalendarSnapshotWorkspaceExecutor,
    CalendarSnapshotWorkspaceVerifier,
    CommunicationDraftExecutor,
    CommunicationDraftVerifier,
    CommunicationsExecutor,
    CommunicationsSendExecutor,
    CommunicationsSendVerifier,
    CommunicationsVerifier,
    DeviceControlExecutor,
    DeviceControlVerifier,
    DeviceSnapshotWorkspaceExecutor,
    DeviceSnapshotWorkspaceVerifier,
    DeviceStatesExecutor,
    DeviceStatesVerifier,
    DocumentsExecutor,
    DocumentsVerifier,
    DocumentWorkspaceExecutor,
    DocumentWorkspaceVerifier,
    HomelabHealthExecutor,
    HomelabHealthVerifier,
    HomelabHealthWorkspaceExecutor,
    HomelabWorkspaceExecutor,
    HomelabWorkspaceVerifier,
    OpenClawGroceryExecutor,
    OpenClawGroceryVerifier,
    OpenClawHomelabExecutor,
    OpenClawHomelabVerifier,
    OpenClawNetworkProbeExecutor,
    OpenClawNetworkProbeVerifier,
    PostgresGroceryListExecutor,
    PostgresGroceryListVerifier,
    ResearchWorkspaceExecutor,
    ResearchWorkspaceVerifier,
    WorkspaceArtifactExecutor,
    WorkspaceArtifactVerifier,
    prepare_reference_action,
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

    def homelab_health_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        return ActionRuntime(
            HomelabHealthExecutor(connection, principal),
            HomelabHealthVerifier(connection, principal),
            {"homelab.read": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def homelab_workspace_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        return ActionRuntime(
            HomelabWorkspaceExecutor(connection, principal),
            HomelabWorkspaceVerifier(principal),
            {
                "homelab.read": frozenset({Role.OWNER}),
                "workspace.write": frozenset({Role.OWNER}),
            },
            prepare=lambda action, current_principal, objective_id: prepare_reference_action(
                action, current_principal, objective_id, connection
            ),
        )

    def homelab_health_workspace_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        return ActionRuntime(
            HomelabHealthWorkspaceExecutor(connection, principal),
            HomelabWorkspaceVerifier(principal),
            {
                "homelab.read": frozenset({Role.OWNER}),
                "workspace.write": frozenset({Role.OWNER}),
            },
            prepare=lambda action, current_principal, objective_id: prepare_reference_action(
                action, current_principal, objective_id, connection
            ),
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

    def workspace_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection
        return ActionRuntime(
            WorkspaceArtifactExecutor(principal),
            WorkspaceArtifactVerifier(principal),
            {"workspace.write": frozenset({Role.OWNER})},
            prepare=prepare_reference_action,
        )

    def calendar_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection, principal
        return ActionRuntime(
            CalendarEventsExecutor(),
            CalendarEventsVerifier(),
            {"calendar.read": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def calendar_create_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection, principal
        provider = configured_calendar_write_provider()
        return ActionRuntime(
            CalendarCreateExecutor(provider),
            CalendarCreateVerifier(provider),
            {"calendar.write": frozenset({Role.OWNER})},
            prepare=prepare_reference_action,
        )

    def calendar_snapshot_workspace_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection
        return ActionRuntime(
            CalendarSnapshotWorkspaceExecutor(principal),
            CalendarSnapshotWorkspaceVerifier(principal),
            {
                "calendar.read": frozenset({Role.OWNER, Role.MEMBER}),
                "workspace.write": frozenset({Role.OWNER}),
            },
            prepare=prepare_reference_action,
        )

    def calendar_communication_draft_runtime(
        connection: Any, principal: Principal
    ) -> ActionRuntime:
        del connection
        return ActionRuntime(
            CalendarCommunicationDraftExecutor(principal),
            CalendarCommunicationDraftVerifier(principal),
            {
                "calendar.read": frozenset({Role.OWNER, Role.MEMBER}),
                "communications.draft": frozenset({Role.OWNER}),
                "workspace.write": frozenset({Role.OWNER}),
            },
            prepare=prepare_reference_action,
        )

    def documents_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection, principal
        return ActionRuntime(
            DocumentsExecutor(),
            DocumentsVerifier(),
            {"documents.read": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def communications_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection, principal
        return ActionRuntime(
            CommunicationsExecutor(),
            CommunicationsVerifier(),
            {"communications.read": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def communications_send_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection, principal
        provider = (
            OpenClawCliCommunicationSendProvider(os.environ["AEGIS_OPENCLAW_MESSAGE_BIN"])
            if os.environ.get("AEGIS_OPENCLAW_MESSAGE_BIN")
            else FixtureCommunicationSendProvider()
        )
        return ActionRuntime(
            CommunicationsSendExecutor(provider),
            CommunicationsSendVerifier(),
            {"communications.send": frozenset({Role.OWNER})},
        )

    def communication_draft_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection
        return ActionRuntime(
            CommunicationDraftExecutor(principal),
            CommunicationDraftVerifier(principal),
            {
                "communications.draft": frozenset({Role.OWNER}),
                "workspace.write": frozenset({Role.OWNER}),
            },
            prepare=prepare_reference_action,
        )

    def devices_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection, principal
        return ActionRuntime(
            DeviceStatesExecutor(),
            DeviceStatesVerifier(),
            {"devices.read": frozenset({Role.OWNER, Role.MEMBER})},
        )

    def device_control_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection
        gateway = (
            HomeAssistantRestControlGateway(
                os.environ["AEGIS_HOME_ASSISTANT_URL"],
                os.environ["AEGIS_HOME_ASSISTANT_TOKEN"],
            )
            if os.environ.get("AEGIS_HOME_ASSISTANT_URL")
            and os.environ.get("AEGIS_HOME_ASSISTANT_TOKEN")
            else FixtureDeviceGateway({})
        )
        return ActionRuntime(
            DeviceControlExecutor(gateway),
            DeviceControlVerifier(gateway, principal),
            {"devices.control": frozenset({Role.OWNER})},
            prepare=prepare_reference_action,
        )

    def device_snapshot_workspace_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection
        return ActionRuntime(
            DeviceSnapshotWorkspaceExecutor(principal),
            DeviceSnapshotWorkspaceVerifier(principal),
            {
                "devices.read": frozenset({Role.OWNER, Role.MEMBER}),
                "workspace.write": frozenset({Role.OWNER}),
            },
            prepare=prepare_reference_action,
        )

    def document_workspace_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection
        return ActionRuntime(
            DocumentWorkspaceExecutor(principal),
            DocumentWorkspaceVerifier(principal),
            {"documents.read": frozenset({Role.OWNER}), "workspace.write": frozenset({Role.OWNER})},
            prepare=prepare_reference_action,
        )

    def research_workspace_runtime(connection: Any, principal: Principal) -> ActionRuntime:
        del connection
        return ActionRuntime(
            ResearchWorkspaceExecutor(principal),
            ResearchWorkspaceVerifier(),
            {"workspace.write": frozenset({Role.OWNER})},
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
        "homelab.service.health": homelab_health_runtime,
        "homelab-reports.inventory.to_workspace": homelab_workspace_runtime,
        "homelab-reports.health.to_workspace": homelab_health_workspace_runtime,
        "network.probe": network_runtime,
        "workspace.artifact.create": workspace_runtime,
        "calendar.events.list": calendar_runtime,
        "calendar.events.create": calendar_create_runtime,
        "calendar-reports.events.snapshot_to_workspace": calendar_snapshot_workspace_runtime,
        "calendar-communications.events.draft": calendar_communication_draft_runtime,
        "documents.list": documents_runtime,
        "communications.messages.list": communications_runtime,
        "communications.messages.send": communications_send_runtime,
        "communication-drafts.messages.draft": communication_draft_runtime,
        "devices.states.list": devices_runtime,
        "device-controls.devices.command.execute": device_control_runtime,
        "device-reports.devices.snapshot_to_workspace": device_snapshot_workspace_runtime,
        "documents.export_to_workspace": document_workspace_runtime,
        "workspace.research_notes.create": research_workspace_runtime,
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
    """Compatibility resolver backed by the same Pack registry as production."""

    from .reference_packs import reference_bundles

    card = next(
        (
            candidate
            for bundle in reference_bundles()
            for candidate in bundle.cards
            if candidate.action.action_id == action_id
        ),
        None,
    )
    if card is None:
        raise LookupError(f"no Pack action card for {action_id}")
    return default_runtime_registry(openclaw_channel).resolve(card, connection, principal)
