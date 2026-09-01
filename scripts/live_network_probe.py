"""Live authorized Network Pack probe through OpenClaw and TCP readback."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

import psycopg

from aegis.audit import PostgresAuditLog
from aegis.contracts import ActionCard, ExecutionRequest, IntentFrame, Principal
from aegis.decoding import StrictDecisionDecoder
from aegis.gateway_rpc import OpenClawWebSocketChannel
from aegis.identity import PostgresSpacePolicy, Role
from aegis.kernel import Kernel
from aegis.network import (
    AuthorizedNetworkScope,
    DiscoveredDevice,
    NetworkScopePolicy,
    PostgresNetworkStore,
)
from aegis.ollama import OllamaHttpTransport, OllamaProvider
from aegis.openclaw import OpenClawExecutor
from aegis.pack_lifecycle import PackManager, PackStatus, PostgresPackStore
from aegis.reference_packs import (
    OpenClawNetworkProbeExecutor,
    OpenClawNetworkProbeVerifier,
    reference_bundles,
)
from aegis.store import PostgresObjectiveStore


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def channel() -> OpenClawWebSocketChannel:
    db = sqlite3.connect(required("AEGIS_OPENCLAW_IDENTITY_DB"))
    row = db.execute(
        "SELECT device_id, private_key_pem, public_key_pem FROM device_identities "
        "WHERE identity_key='primary'"
    ).fetchone()
    db.close()
    if row is None:
        raise SystemExit("OpenClaw primary device identity was not found")
    return OpenClawWebSocketChannel(
        required("AEGIS_OPENCLAW_GATEWAY_URL"),
        required("AEGIS_OPENCLAW_TOKEN"),
        timeout=15,
        persistent=True,
        device_id=str(row[0]),
        device_token=required("AEGIS_OPENCLAW_DEVICE_TOKEN"),
        private_key_pem=str(row[1]),
        public_key_pem=str(row[2]),
    )


class RuntimePolicy:
    def allows(self, request: ExecutionRequest) -> bool:
        return request.action.action_id == "network.probe"


class NoApproval:
    def required(self, request: object) -> bool:
        return False

    def approved(self, request: object) -> bool:
        return True


def main() -> None:
    port = int(os.environ.get("AEGIS_NETWORK_TEST_PORT", "18084"))
    root = Path(f"/tmp/aegis-network-alpha-{os.getpid()}")
    root.mkdir(mode=0o700)
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    conn = psycopg.connect(required("AEGIS_DATABASE_URL"))
    sock = channel()
    principal = Principal(
        id=os.environ.get("AEGIS_PRINCIPAL_ID", "alice"),
        vault_id=os.environ.get("AEGIS_VAULT_ID", "alice-vault"),
        space_ids=(os.environ.get("AEGIS_SPACE_ID", "apartment"),),
    )
    try:
        network = PostgresNetworkStore(conn)
        network.save_scope(
            principal,
            AuthorizedNetworkScope("alpha-lab", ("127.0.0.0/8",), "owned local acceptance lab"),
        )
        network.save_device(principal, DiscoveredDevice("127.0.0.1", "localhost", ("http",)))
        manager = PackManager(store=PostgresPackStore(conn))
        for bundle in reference_bundles():
            try:
                manager.status(bundle.manifest.pack_id)
            except KeyError:
                manager.discover(bundle)
        if manager.status("network") is PackStatus.DISCOVERED:
            manager.install("network", frozenset({"network.read"}))
            manager.enable("network")
        card = next(
            card for card in manager.retrieve("network") if card.action.action_id == "network.probe"
        )
        card = ActionCard(
            action=card.action.model_copy(
                update={
                    "arguments": {
                        "address": "127.0.0.1",
                        "port": port,
                        "scope_id": "alpha-lab",
                    }
                }
            ),
            summary=card.summary,
            relevance=card.relevance,
        )
        base_policy = PostgresSpacePolicy(
            conn, {"network.read": frozenset({Role.OWNER, Role.MEMBER})}
        )
        kernel = Kernel(
            OllamaProvider(
                os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b"),
                OllamaHttpTransport(required("AEGIS_OLLAMA_URL"), timeout=120),
                max_repairs=0,
            ),
            StrictDecisionDecoder(),
            NetworkScopePolicy(base_policy, network),
            OpenClawExecutor(OpenClawNetworkProbeExecutor(sock), RuntimePolicy(), NoApproval()),
            OpenClawNetworkProbeVerifier(),
            store=PostgresObjectiveStore(conn),
            audit=PostgresAuditLog(conn),
        )
        result = kernel.run(
            IntentFrame(
                principal=principal,
                utterance=(
                    "Probe the authorized local alpha network target and verify it is reachable."
                ),
                correlation_id=UUID(os.environ.get("AEGIS_CORRELATION_ID", str(uuid4()))),
            ),
            (card,),
        )
        print(result.model_dump_json(indent=2))
    finally:
        sock.close()
        conn.close()
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
        root.rmdir()


if __name__ == "__main__":
    main()
