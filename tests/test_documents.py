import pytest

from aegis.documents import configured_document_provider


def test_configured_document_provider_is_empty_by_default(monkeypatch):
    monkeypatch.delenv("AEGIS_DOCUMENT_FIXTURE_JSON", raising=False)
    assert configured_document_provider().list_documents() == ()


def test_configured_document_provider_is_bounded_and_explicit(monkeypatch):
    monkeypatch.setenv(
        "AEGIS_DOCUMENT_FIXTURE_JSON",
        '[{"document_id":"guide","title":"Guide","text":"Use the safe path."}]',
    )
    document = configured_document_provider().list_documents()[0]
    assert document.document_id == "guide"
    assert document.source == "configured_document_fixture"


def test_configured_document_provider_rejects_malformed_fixture(monkeypatch):
    monkeypatch.setenv("AEGIS_DOCUMENT_FIXTURE_JSON", "{}")
    with pytest.raises(ValueError):
        configured_document_provider()
