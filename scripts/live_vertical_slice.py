"""Run the live Qwen -> PostgreSQL -> OpenClaw grocery acceptance path.

All endpoints and credentials are supplied by environment variables. The
script intentionally uses the existing Kitchen ActionCard and Core pipeline;
it does not replace live dependencies with fakes.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import psycopg

from aegis.audit import PostgresAuditLog
from aegis.contracts import (
    ActionCard,
    ExecutionRequest,
    IntentFrame,
    Principal,
)
from aegis.decoding import StrictDecisionDecoder
from aegis.gateway_rpc import OpenClawWebSocketChannel
from aegis.household import PostgresHouseholdStore
from aegis.identity import PostgresSpacePolicy, Role
from aegis.kernel import Kernel
from aegis.ollama import OllamaHttpTransport, OllamaProvider
from aegis.openclaw import OpenClawExecutor
from aegis.pack_lifecycle import PackManager, PackStatus, PostgresPackStore
from aegis.reference_packs import (
    OpenClawGroceryExecutor,
    OpenClawGroceryVerifier,
    reference_bundles,
)
from aegis.store import PostgresObjectiveStore


class AcceptanceRuntime:
    def allows(self, request: ExecutionRequest) -> bool:
        return request.action.action_id == "kitchen.groceries.add"


class NoApproval:
    def required(self, request: object) -> bool:
        return False

    def approved(self, request: object) -> bool:
        return True


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def device_identity() -> tuple[str, str, str, str]:
    identity_db = required("AEGIS_OPENCLAW_IDENTITY_DB")
    row = sqlite3.connect(identity_db).execute(
        "select device_id, private_key_pem, public_key_pem "
        "from device_identities where identity_key='primary'"
    ).fetchone()
    if row is None:
        raise SystemExit("OpenClaw identity database has no primary device identity")
    return (
        str(row[0]),
        required("AEGIS_OPENCLAW_DEVICE_TOKEN"),
        str(row[1]),
        str(row[2]),
    )


def open_channel() -> OpenClawWebSocketChannel:
    device_id, device_token, private_key, public_key = device_identity()
    return OpenClawWebSocketChannel(
        required("AEGIS_OPENCLAW_GATEWAY_URL"),
        required("AEGIS_OPENCLAW_TOKEN"),
        timeout=15,
        persistent=True,
        device_id=device_id,
        device_token=device_token,
        private_key_pem=private_key,
        public_key_pem=public_key,
    )


def seed_fixture(connection: psycopg.Connection[object]) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into spaces (id, name) values ('apartment', 'Apartment') "
            "on conflict (id) do nothing"
        )
        cursor.execute(
            "insert into space_memberships (principal_id, space_id, role, active) "
            "values ('alice', 'apartment', 'owner', true) "
            "on conflict (principal_id, space_id) do update set active=true"
        )
    connection.commit()


def run_once(correlation: UUID, path: str) -> object:
    connection = psycopg.connect(required("AEGIS_DATABASE_URL"))
    channel = open_channel()
    try:
        manager = PackManager(store=PostgresPackStore(connection))
        for bundle in reference_bundles():
            try:
                manager.status(bundle.manifest.pack_id)
            except KeyError:
                manager.discover(bundle)
        if manager.status("kitchen") is not PackStatus.ENABLED:
            if manager.status("kitchen") is PackStatus.DISCOVERED:
                manager.install("kitchen", frozenset({"kitchen.write"}))
            manager.enable("kitchen")
        card_action = manager.retrieve("kitchen")[0].action.model_copy(
            update={"arguments": {"item": "rice"}}
        )
        card = ActionCard(action=card_action, summary="Add an item to groceries", relevance=1)
        kernel = Kernel(
            OllamaProvider(
                required("AEGIS_OLLAMA_MODEL"),
                OllamaHttpTransport(required("AEGIS_OLLAMA_URL"), timeout=120),
                max_repairs=0,
            ),
            StrictDecisionDecoder(),
            PostgresSpacePolicy(
                connection,
                {"kitchen.write": frozenset({Role.OWNER, Role.MEMBER})},
            ),
            OpenClawExecutor(
                OpenClawGroceryExecutor(
                    channel,
                    path,
                    canonical_store=PostgresHouseholdStore(connection),
                    principal=Principal(
                        id="alice", vault_id="alice-vault", space_ids=("apartment",)
                    ),
                ),
                AcceptanceRuntime(),
                NoApproval(),
            ),
            OpenClawGroceryVerifier(
                canonical_store=PostgresHouseholdStore(connection),
                principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            ),
            store=PostgresObjectiveStore(connection),
            audit=PostgresAuditLog(connection),
        )
        return kernel.run(
            IntentFrame(
                principal=Principal(
                    id="alice", vault_id="alice-vault", space_ids=("apartment",)
                ),
                utterance="Add rice to groceries.",
                correlation_id=correlation,
            ),
            (card,),
        )
    finally:
        channel.close()
        connection.close()


def main() -> None:
    path = required("AEGIS_LIVE_GROCERY_PATH")
    connection = psycopg.connect(required("AEGIS_DATABASE_URL"))
    try:
        seed_fixture(connection)
    finally:
        connection.close()
    correlation = UUID(os.environ.get("AEGIS_CORRELATION_ID", str(uuid4())))
    first = run_once(correlation, path)
    replay = run_once(correlation, path)
    records = Path(path).read_text(encoding="utf-8").splitlines()
    matching = [line for line in records if line.startswith(f"{correlation}:")]
    print(
        json.dumps(
            {
                "correlation_id": str(correlation),
                "first": first.model_dump(mode="json"),  # type: ignore[attr-defined]
                "replay": replay.model_dump(mode="json"),  # type: ignore[attr-defined]
                "matching_external_records": len(matching),
                "external_records": records,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
