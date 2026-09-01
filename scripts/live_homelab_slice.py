"""Live bounded Homelab Pack acceptance through OpenClaw and HTTP health readback."""

from __future__ import annotations

import os
import shlex
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
from aegis.ollama import OllamaHttpTransport, OllamaProvider
from aegis.openclaw import OpenClawExecutor
from aegis.pack_lifecycle import PackManager, PackStatus, PostgresPackStore
from aegis.reference_packs import (
    OpenClawHomelabExecutor,
    OpenClawHomelabVerifier,
    reference_bundles,
)
from aegis.store import PostgresObjectiveStore


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def open_channel() -> OpenClawWebSocketChannel:
    identity_db = required("AEGIS_OPENCLAW_IDENTITY_DB")
    row = sqlite3.connect(identity_db).execute(
        "SELECT device_id, private_key_pem, public_key_pem FROM device_identities "
        "WHERE identity_key='primary'"
    ).fetchone()
    if row is None:
        raise SystemExit("OpenClaw identity database has no primary device identity")
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
        return request.action.action_id == "homelab.service.restart"


class NoApproval:
    def required(self, request: object) -> bool:
        return False

    def approved(self, request: object) -> bool:
        return True


def main() -> None:
    port = int(os.environ.get("AEGIS_HOMELAB_TEST_PORT", "18081"))
    root = Path(f"/tmp/aegis-homelab-alpha-{os.getpid()}")
    root.mkdir(mode=0o700)
    pid_file = root / "service.pid"
    log_file = root / "service.log"
    service_command = (
        f"kill $(cat {shlex.quote(str(pid_file))}) 2>/dev/null || true; "
        f"nohup setsid {shlex.quote(sys.executable)} -m http.server {port} "
        f"--bind 127.0.0.1 --directory {shlex.quote(str(root))} "
        f">{shlex.quote(str(log_file))} 2>&1 & echo $! > {shlex.quote(str(pid_file))}; "
        "sleep 1"
    )
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
        stdout=log_file.open("w"),
        stderr=subprocess.STDOUT,
    )
    pid_file.write_text(str(server.pid), encoding="utf-8")
    time.sleep(0.5)
    connection = psycopg.connect(required("AEGIS_DATABASE_URL"))
    channel = open_channel()
    principal = Principal(
        id=os.environ.get("AEGIS_PRINCIPAL_ID", "alice"),
        vault_id=os.environ.get("AEGIS_VAULT_ID", "alice-vault"),
        space_ids=(os.environ.get("AEGIS_SPACE_ID", "apartment"),),
    )
    try:
        manager = PackManager(store=PostgresPackStore(connection))
        for bundle in reference_bundles():
            try:
                manager.status(bundle.manifest.pack_id)
            except KeyError:
                manager.discover(bundle)
        if manager.status("homelab") is PackStatus.DISCOVERED:
            manager.install("homelab", frozenset({"homelab.service.restart"}))
            manager.enable("homelab")
        card = next(
            card for card in manager.retrieve("homelab")
            if card.action.action_id == "homelab.service.restart"
        )
        card = ActionCard(
            action=card.action.model_copy(update={"arguments": {"service": "alpha-http"}}),
            summary=card.summary,
            relevance=card.relevance,
        )
        executor = OpenClawHomelabExecutor(channel, {"alpha-http": service_command})
        verifier = OpenClawHomelabVerifier({"alpha-http": f"http://127.0.0.1:{port}/"})
        kernel = Kernel(
            OllamaProvider(
                os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b"),
                OllamaHttpTransport(required("AEGIS_OLLAMA_URL"), timeout=120),
                max_repairs=0,
            ),
            StrictDecisionDecoder(),
            PostgresSpacePolicy(
                connection,
                {"homelab.service.restart": frozenset({Role.OWNER, Role.MEMBER})},
            ),
            OpenClawExecutor(executor, RuntimePolicy(), NoApproval()),
            verifier,
            store=PostgresObjectiveStore(connection),
            audit=PostgresAuditLog(connection),
        )
        result = kernel.run(
            IntentFrame(
                principal=principal,
                utterance="Restart the authorized alpha HTTP service and verify it is healthy.",
                correlation_id=UUID(os.environ.get("AEGIS_CORRELATION_ID", str(uuid4()))),
            ),
            (card,),
        )
        print(result.model_dump_json(indent=2))
    finally:
        channel.close()
        connection.close()
        try:
            server.terminate()
            server.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            server.kill()
        pid_file.unlink(missing_ok=True)
        log_file.unlink(missing_ok=True)
        root.rmdir()


if __name__ == "__main__":
    main()
