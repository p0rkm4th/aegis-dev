from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from aegis.audit import PostgresAuditLog


@pytest.mark.skipif(
    not os.environ.get("AEGIS_TEST_DATABASE_URL"), reason="requires disposable PostgreSQL"
)
def test_postgres_audit_chain_serializes_fifty_connections():
    import psycopg

    url = os.environ["AEGIS_TEST_DATABASE_URL"]
    suffix = uuid4().hex
    event_type = f"test.audit.concurrent.{suffix}"
    principals = [f"audit-test-{suffix}-{i}" for i in range(50)]
    setup = psycopg.connect(url)
    for principal in principals:
        setup.execute(
            "INSERT INTO aegis_principals (id, external_subject) VALUES (%s, %s)",
            (principal, principal),
        )
    setup.commit()
    setup.close()

    def append(principal: str) -> str:
        connection = psycopg.connect(url)
        try:
            return (
                PostgresAuditLog(connection)
                .append(event_type, principal, {"test_run": suffix})
                .event_hash
            )
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=50) as pool:
        hashes = list(pool.map(append, principals))

    check = psycopg.connect(url)
    try:
        count = check.execute(
            "SELECT count(*) FROM audit_events WHERE event_type = %s", (event_type,)
        ).fetchone()[0]
        forks = check.execute(
            "SELECT previous_hash FROM audit_events WHERE event_type = %s "
            "GROUP BY previous_hash HAVING count(*) > 1",
            (event_type,),
        ).fetchall()
        assert count == 50
        assert len(set(hashes)) == 50
        assert forks == []
        assert PostgresAuditLog(check).verify()
    finally:
        check.close()
