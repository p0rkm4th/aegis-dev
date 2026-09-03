from __future__ import annotations

import pytest

from aegis.research import (
    DocumentFetcher,
    EvidenceSet,
    FetchedDocument,
    ResearchService,
    SearchCandidate,
    SearchRequest,
    TrafilaturaContentExtractor,
    _validated_url,
    resolve_public,
)


def test_research_request_is_bounded() -> None:
    with pytest.raises(ValueError):
        SearchRequest(" ")
    with pytest.raises(ValueError):
        SearchRequest("x", limit=6)
    with pytest.raises(ValueError):
        SearchRequest("x" * 501)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.test/a",
        "http://user:pass@example.test/a",
        "javascript:alert(1)",
    ],
)
def test_research_fetch_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(ValueError):
        _validated_url(url)


@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "10.0.0.1", "169.254.169.254", "fc00::1"])
def test_research_fetch_rejects_non_global_destinations(address: str) -> None:
    with pytest.raises(ValueError):
        DocumentFetcher(resolver=lambda _host, _port: (address,)).fetch("http://example.test/")


def test_research_fetch_validates_all_resolved_addresses() -> None:
    with pytest.raises(ValueError):
        DocumentFetcher(resolver=lambda _host, _port: ("93.184.216.34", "192.168.1.2")).fetch(
            "http://example.test/"
        )


def test_public_resolver_rejects_private_resolution() -> None:
    with pytest.raises(ValueError):
        resolve_public("localhost", 80)


def test_trafilatura_extractor_never_fetches_network_data(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    captured: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "trafilatura",
        types.SimpleNamespace(
            extract=lambda value, **_kwargs: captured.append(value) or "Readable"
        ),
    )
    from aegis.research import FetchedDocument

    assert (
        TrafilaturaContentExtractor().extract(
            FetchedDocument("https://example.test", "text/html", b"<p>hello</p>")
        )
        == "Readable"
    )
    assert captured == ["<p>hello</p>"]


def test_research_service_deduplicates_and_returns_bounded_untrusted_evidence() -> None:
    class Provider:
        provider_id = "fake"

        def search(self, _request: SearchRequest) -> tuple[SearchCandidate, ...]:
            return (
                SearchCandidate("One", "https://public.example/one"),
                SearchCandidate("Duplicate", "https://public.example/one#fragment"),
            )

    class Fetcher:
        def fetch(self, url: str) -> FetchedDocument:
            return FetchedDocument(url, "text/plain", b"untrusted page text")

    class Extractor:
        def extract(self, document: FetchedDocument) -> str:
            return document.body.decode()

    result = ResearchService(Provider(), Fetcher(), Extractor()).collect(SearchRequest("public"))

    assert isinstance(result, EvidenceSet)
    assert len(result.evidence) == 1
    assert result.evidence[0].text == "untrusted page text"
