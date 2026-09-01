"""Human-facing alpha adapter for the existing AEGIS semantic pipeline."""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import psycopg

from .audit import PostgresAuditLog
from .contracts import ActionCard, IntentFrame, Principal
from .decoding import StrictDecisionDecoder
from .embeddings import OllamaEmbeddingProvider, PostgresMemoryVectorIndex
from .finance import FinanceLedger, FinanceReadFastPath, PostgresFinanceSnapshotStore
from .gateway_rpc import OpenClawWebSocketChannel
from .household import (
    HouseholdObligation,
    HouseholdReadFastPath,
    PostgresChoreExecutor,
    PostgresChoreVerifier,
    PostgresEventExecutor,
    PostgresEventVerifier,
    PostgresHouseholdStore,
)
from .identity import KeycloakIdentityProvider, KeycloakOIDCClient, PostgresSpacePolicy, Role
from .kernel import Kernel
from .ollama import OllamaHttpTransport, OllamaProvider
from .openclaw import OpenClawExecutor
from .pack_lifecycle import PackManager, PackStatus, PostgresPackStore
from .personal import PersonalMemoryFastPath, PostgresPersonalStateStore
from .projections import SharedObligation
from .reference_packs import (
    OpenClawGroceryExecutor,
    OpenClawGroceryVerifier,
    PostgresGroceryListExecutor,
    PostgresGroceryListVerifier,
    reference_bundles,
)
from .store import PostgresObjectiveStore
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


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _principal() -> Principal:
    token = os.environ.get("AEGIS_KEYCLOAK_ACCESS_TOKEN")
    if token:
        issuer = _required("AEGIS_KEYCLOAK_ISSUER")
        return KeycloakOIDCClient(issuer).principal_from_access_token(token)
    # Local alpha mode still crosses the identity mapping boundary. It uses an
    # explicitly configured development identity rather than pretending to be a
    # bearer-token session.
    claims = {
        "sub": os.environ.get("AEGIS_PRINCIPAL_ID", "alice"),
        "aegis_vault_id": os.environ.get("AEGIS_VAULT_ID", "alice-vault"),
        "aegis_space_ids": [os.environ.get("AEGIS_SPACE_ID", "apartment")],
    }
    return KeycloakIdentityProvider().principal_from_claims(claims)


def _apply_migrations(connection: Any) -> None:
    if os.environ.get("AEGIS_AUTO_MIGRATE", "1").lower() in {"0", "false", "no"}:
        return
    root = Path(__file__).resolve().parents[2]
    for migration in sorted((root / "migrations").glob("*.sql")):
        connection.execute(migration.read_text(encoding="utf-8"))
    connection.commit()


def _ensure_local_identity(connection: Any, principal: Principal) -> None:
    space_id = principal.space_ids[0]
    connection.execute(
        "INSERT INTO aegis_principals (id, external_subject) VALUES (%s, %s) "
        "ON CONFLICT (id) DO NOTHING",
        (principal.id, principal.id),
    )
    connection.execute(
        "INSERT INTO vaults (id, owner_principal_id) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
        (principal.vault_id, principal.id),
    )
    connection.execute(
        "INSERT INTO spaces (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
        (space_id, os.environ.get("AEGIS_SPACE_NAME", "Apartment")),
    )
    connection.execute(
        "INSERT INTO space_memberships (principal_id, space_id, role, active) "
        "VALUES (%s, %s, 'owner', TRUE) ON CONFLICT (principal_id, space_id) "
        "DO NOTHING",
        (principal.id, space_id),
    )
    connection.commit()


def _openclaw_channel() -> OpenClawWebSocketChannel:
    identity_db = _required("AEGIS_OPENCLAW_IDENTITY_DB")
    row = (
        sqlite3.connect(identity_db)
        .execute(
            "SELECT device_id, private_key_pem, public_key_pem FROM device_identities "
            "WHERE identity_key='primary'"
        )
        .fetchone()
    )
    if row is None:
        raise RuntimeError("OpenClaw primary device identity was not found")
    return OpenClawWebSocketChannel(
        _required("AEGIS_OPENCLAW_GATEWAY_URL"),
        _required("AEGIS_OPENCLAW_TOKEN"),
        timeout=15,
        persistent=True,
        device_id=str(row[0]),
        device_token=_required("AEGIS_OPENCLAW_DEVICE_TOKEN"),
        private_key_pem=str(row[1]),
        public_key_pem=str(row[2]),
    )


def _domain_and_action(utterance: str, manager: PackManager) -> tuple[str, ActionCard]:
    text = utterance.lower()
    if "task" in text:
        domain = "tasks"
        if "event" in text:
            action_id = "tasks.events.create"
        elif "chore" in text:
            action_id = (
                "tasks.chores.list"
                if text.startswith(("what", "show", "list"))
                else "tasks.chores.create"
            )
        else:
            action_id = (
                "tasks.list" if text.startswith(("what", "show", "list")) else "tasks.create"
            )
    elif "chore" in text:
        domain = "tasks"
        action_id = (
            "tasks.chores.list"
            if text.startswith(("what", "show", "list"))
            else "tasks.chores.create"
        )
    elif "event" in text or "inspection" in text:
        domain = "tasks"
        action_id = "tasks.events.create"
    elif any(word in text for word in ("grocery", "groceries", "rice", "food")):
        domain = "kitchen"
        action_id = (
            "kitchen.groceries.list"
            if text.startswith(("what", "show", "list"))
            else "kitchen.groceries.add"
        )
    else:
        raise ValueError("alpha supports groceries and tasks; try one of the four demo requests")
    cards = manager.retrieve(domain)
    card = next((item for item in cards if item.action.action_id == action_id), None)
    if card is None:
        raise RuntimeError(f"enabled Pack did not provide ActionCard {action_id}")
    action = card.action
    if action_id == "kitchen.groceries.add":
        match = re.search(r"add\s+(.+?)\s+to\s+(?:the\s+)?grocer(?:y|ies)\b", text)
        if match is None:
            raise ValueError("tell AEGIS what to add, for example: Add rice to groceries.")
        action = action.model_copy(update={"arguments": {"item": match.group(1).strip()}})
    elif action_id == "tasks.create":
        match = re.search(r"(?:create\s+)?(?:a\s+)?task\s+(?:to\s+)?(.+)$", text)
        if match is None:
            raise ValueError("tell AEGIS the task, for example: Create a task to buy cat food.")
        action = action.model_copy(update={"arguments": {"title": match.group(1).strip()}})
    elif action_id == "tasks.chores.create":
        match = re.search(r"(?:create|add)\s+(?:a\s+)?chore\s+(?:to\s+)?(.+)$", text)
        if match is None:
            raise ValueError(
                "tell AEGIS the chore, for example: Create a chore to clean the kitchen."
            )
        action = action.model_copy(update={"arguments": {"title": match.group(1).strip()}})
    elif action_id == "tasks.events.create":
        match = re.search(r"(?:create|add)\s+(?:an?\s+)?event\s+(?:for\s+)?(.+)$", text)
        if match is None:
            raise ValueError(
                "tell AEGIS the event, for example: Create an event for apartment inspection."
            )
        title = match.group(1).strip()
        if title.endswith(" tomorrow"):
            title = title.removesuffix(" tomorrow").strip()
            starts_at = datetime.now(timezone.utc) + timedelta(days=1)
        else:
            starts_at = datetime.now(timezone.utc)
        action = action.model_copy(
            update={"arguments": {"title": title, "starts_at": starts_at.isoformat()}}
        )
    return domain, ActionCard(action=action, summary=card.summary, relevance=card.relevance)


def _format(result: Any) -> str:
    if result.state.value != "completed":
        return f"Not completed — {result.message}"
    evidence = result.evidence
    if evidence.get("canonical_items") is not None:
        items = evidence["canonical_items"]
        return "Groceries: " + (", ".join(items) if items else "(empty)")
    if evidence.get("canonical_tasks") is not None:
        tasks = evidence["canonical_tasks"]
        listing = "; ".join(f"{item['title']} ({item['status']})" for item in tasks)
        return "Tasks: " + (listing if tasks else "(empty)")
    if evidence.get("memories") is not None:
        memories = evidence["memories"]
        if not memories:
            return "Memories: (none found)"
        return "Memories: " + "; ".join(
            f"{item['content']} [{item['provenance']}]" for item in memories
        )
    if evidence.get("projects") is not None:
        projects = evidence["projects"]
        return "Projects: " + (
            "; ".join(item["name"] for item in projects) if projects else "(none)"
        )
    if evidence.get("goals") is not None:
        goals = evidence["goals"]
        return "Goals: " + (
            "; ".join(
                f"{item['description']}" + (f" [{item['project']}]" if item["project"] else "")
                for item in goals
            )
            if goals
            else "(none)"
        )
    if evidence.get("obligations") is not None:
        obligations = evidence["obligations"]
        outstanding = [item for item in obligations if not item["settled"]]
        return "Outstanding obligations: " + (
            "; ".join(f"{item['title']} ({item['responsible_id']})" for item in outstanding)
            if outstanding
            else "(none)"
        )
    if evidence.get("chores") is not None:
        chores = evidence["chores"]
        return "Chores: " + (
            "; ".join(
                f"{item['title']} ({item['assignee_id']})"
                for item in chores
                if not item["completed"]
            )
            if chores
            else "(none)"
        )
    if evidence.get("events") is not None:
        events = evidence["events"]
        return "Events: " + ("; ".join(item["title"] for item in events) if events else "(none)")
    if evidence.get("affordable") is not None:
        status = "yes" if evidence["affordable"] else "no"
        return (
            f"Affordable: {status} (purchase ${evidence['purchase_cents'] / 100:.2f}; "
            f"shared obligations ${evidence['shared_obligations_cents'] / 100:.2f})"
        )
    if evidence.get("collection") == "chores" and evidence.get("title"):
        return f"Done — created chore: {evidence['title']}"
    if evidence.get("collection") == "events" and evidence.get("title"):
        return f"Done — created event: {evidence['title']}"
    if evidence.get("title"):
        return f"Done — created task: {evidence['title']}"
    if evidence.get("item"):
        return f"Done — added {evidence['item']} to groceries"
    return f"Done — {result.message}"


def handle(utterance: str, principal: Principal) -> str:
    connection = psycopg.connect(_required("AEGIS_DATABASE_URL"))
    channel: OpenClawWebSocketChannel | None = None
    try:
        _apply_migrations(connection)
        if not os.environ.get("AEGIS_KEYCLOAK_ACCESS_TOKEN"):
            _ensure_local_identity(connection, principal)
        intent = IntentFrame(principal=principal, utterance=utterance)
        household_store = PostgresHouseholdStore(connection)
        if FinanceReadFastPath.matches(utterance):
            snapshot = household_store.read_snapshot(principal)
            household_obligations = cast(
                tuple[HouseholdObligation, ...], snapshot.get("obligations", ())
            )
            obligations = tuple(
                SharedObligation(item.title, item.amount)
                for item in household_obligations
                if not item.settled
            )
            finance_result = FinanceReadFastPath(
                FinanceLedger(PostgresFinanceSnapshotStore(connection))
            ).resolve(intent, obligations)
            if finance_result is not None:
                PostgresAuditLog(connection).append(
                    "finance.affordability.read",
                    principal.id,
                    {
                        "purchase_cents": finance_result.evidence["purchase_cents"],
                        "shared_obligations_cents": finance_result.evidence[
                            "shared_obligations_cents"
                        ],
                        "affordable": finance_result.evidence["affordable"],
                    },
                )
                return _format(finance_result)
        if HouseholdReadFastPath.matches(utterance):
            household_result = HouseholdReadFastPath(
                household_store.read_snapshot(principal)
            ).resolve(intent)
            if household_result is not None:
                return _format(household_result)
        personal_state = PostgresPersonalStateStore(connection, principal.vault_id).load()
        semantic_enabled = os.environ.get("AEGIS_SEMANTIC_MEMORY", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if semantic_enabled:
            embedding_provider = OllamaEmbeddingProvider(
                os.environ.get("AEGIS_EMBEDDING_MODEL", "nomic-embed-text"),
                _required("AEGIS_OLLAMA_URL"),
            )
            vector_index = PostgresMemoryVectorIndex(connection)
            embeddings = embedding_provider.embed(
                tuple(memory.content for memory in personal_state.memories.values())
            )
            for memory, embedding in zip(personal_state.memories.values(), embeddings):
                vector_index.upsert(
                    principal.vault_id, memory.memory_id, embedding, embedding_provider.model
                )
            connection.commit()
            memory_fast_path = PersonalMemoryFastPath(
                personal_state,
                embedding_provider=embedding_provider,
                vector_index=vector_index,
                vault_id=principal.vault_id,
            )
        else:
            memory_fast_path = PersonalMemoryFastPath(personal_state)
        memory_result = memory_fast_path.resolve(intent)
        if memory_result is not None:
            return _format(memory_result)
        manager = PackManager(store=PostgresPackStore(connection))
        for bundle in reference_bundles():
            try:
                manager.status(bundle.manifest.pack_id)
                installed_ids = {
                    card.action.action_id
                    for card in manager._bundles[bundle.manifest.pack_id].cards
                }
                required_ids = {card.action.action_id for card in bundle.cards}
                if not required_ids.issubset(installed_ids):
                    manager.remove(bundle.manifest.pack_id)
                    manager.discover(bundle)
            except KeyError:
                manager.discover(bundle)
        for pack_id in ("tasks", "kitchen"):
            if manager.status(pack_id) is PackStatus.DISCOVERED:
                manager.install(pack_id, frozenset(manager._bundles[pack_id].manifest.permissions))
                manager.enable(pack_id)
            elif manager.status(pack_id) is PackStatus.INSTALLED:
                manager.enable(pack_id)
        domain, card = _domain_and_action(utterance, manager)
        principal_store = PostgresHouseholdStore(connection)
        task_store = PostgresTaskStore(connection)
        if card.action.action_id == "kitchen.groceries.add":
            channel = _openclaw_channel()
            executor: Any = OpenClawExecutor(
                OpenClawGroceryExecutor(
                    channel,
                    os.environ.get("AEGIS_LIVE_GROCERY_PATH", "/tmp/aegis-alpha-groceries.tsv"),
                    principal_store,
                    principal,
                ),
                _RuntimePolicy(),
                _NoApproval(),
            )
            verifier: Any = OpenClawGroceryVerifier(principal_store, principal)
            permissions = {"kitchen.write": frozenset({Role.OWNER, Role.MEMBER})}
        elif card.action.action_id == "kitchen.groceries.list":
            executor = PostgresGroceryListExecutor(principal_store, principal)
            verifier = PostgresGroceryListVerifier(principal_store, principal)
            permissions = {"kitchen.read": frozenset({Role.OWNER, Role.MEMBER})}
        elif card.action.action_id == "tasks.create":
            executor = PostgresTaskExecutor(task_store, principal)
            verifier = PostgresTaskVerifier(task_store, principal)
            permissions = {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})}
        elif card.action.action_id == "tasks.chores.create":
            executor = PostgresChoreExecutor(principal_store, principal)
            verifier = PostgresChoreVerifier(principal_store, principal)
            permissions = {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})}
        elif card.action.action_id == "tasks.events.create":
            executor = PostgresEventExecutor(principal_store, principal)
            verifier = PostgresEventVerifier(principal_store, principal)
            permissions = {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})}
        else:
            executor = PostgresTaskListExecutor(task_store, principal)
            verifier = PostgresTaskListVerifier(task_store, principal)
            permissions = {"tasks.read": frozenset({Role.OWNER, Role.MEMBER})}
        kernel = Kernel(
            OllamaProvider(
                os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b"),
                OllamaHttpTransport(_required("AEGIS_OLLAMA_URL")),
            ),
            StrictDecisionDecoder(),
            PostgresSpacePolicy(connection, permissions),
            executor,
            verifier,
            store=PostgresObjectiveStore(connection),
            audit=PostgresAuditLog(connection),
        )
        result = kernel.run(intent, (card,))
        return _format(result)
    finally:
        if channel is not None:
            channel.close()
        connection.close()


def main() -> None:
    principal = _principal()
    print("AEGIS alpha. Type a request, or 'quit' to exit.")
    while True:
        try:
            utterance = input("aegis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if utterance.lower() in {"quit", "exit"}:
            return
        if not utterance:
            continue
        try:
            print(_format_error(handle(utterance, principal)))
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"Not completed — {exc}")


def _format_error(message: str) -> str:
    return message


if __name__ == "__main__":
    main()
