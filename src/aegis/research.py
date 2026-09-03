"""Bounded, non-authoritative external research ports.

Research is deliberately an answer-only integration.  Search results and
fetched documents are untrusted evidence; they never enter action decoding,
authorization, or canonical state.
"""

from __future__ import annotations

import ipaddress
import json
import socket
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from http.client import HTTPConnection, HTTPSConnection
from importlib import import_module
from ssl import create_default_context
from typing import Callable, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import urlopen

MAX_QUERY_LENGTH = 500
MAX_CANDIDATES = 5
MAX_REDIRECTS = 3
MAX_RESPONSE_BYTES = 1_000_000
MAX_EXTRACTED_TEXT = 40_000


class KnowledgeSource(StrEnum):
    """Evidence provenance choices; none of these are canonical truth."""

    GENERAL_MODEL = "general_model_knowledge"
    EXTERNAL = "external_evidence"
    MIXED = "mixed_evidence"


@dataclass(frozen=True)
class SearchRequest:
    query: str
    limit: int = 5

    def __post_init__(self) -> None:
        query = self.query.strip()
        if not query or len(query) > MAX_QUERY_LENGTH:
            raise ValueError("research query must be non-empty and bounded")
        if not 1 <= self.limit <= MAX_CANDIDATES:
            raise ValueError("research result limit is out of bounds")


@dataclass(frozen=True)
class SearchCandidate:
    title: str
    url: str
    snippet: str | None = None
    published_at: str | None = None


class SearchProvider(Protocol):
    provider_id: str

    def search(self, request: SearchRequest) -> tuple[SearchCandidate, ...]: ...


@dataclass(frozen=True)
class Evidence:
    source_id: str
    final_url: str
    title: str
    text: str
    retrieved_at: datetime
    published_at: str | None = None
    snippet: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.final_url or not self.title:
            raise ValueError("evidence identity is required")
        if not self.text.strip() or len(self.text) > MAX_EXTRACTED_TEXT:
            raise ValueError("evidence text is empty or exceeds the bound")


@dataclass(frozen=True)
class EvidenceSet:
    query: str
    provider_id: str
    evidence: tuple[Evidence, ...]
    retrieved_at: datetime

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.provider_id:
            raise ValueError("evidence set identity is required")
        if not 1 <= len(self.evidence) <= MAX_CANDIDATES:
            raise ValueError("evidence set must contain a bounded usable source set")


@dataclass(frozen=True)
class ResearchAnswer:
    """Answer-only output with truthful, non-authoritative provenance."""

    text: str
    source_kind: KnowledgeSource
    evidence: EvidenceSet
    authoritative: bool = False

    def __post_init__(self) -> None:
        if not self.text.strip() or self.authoritative:
            raise ValueError("research answers must be non-empty and non-authoritative")


class AnswerSynthesizer(Protocol):
    """A provider with no action cards, tools, or plan proposal surface."""

    def synthesize(
        self, question: str, evidence: EvidenceSet, local_context: dict[str, object] | None = None
    ) -> str: ...


@dataclass(frozen=True)
class FetchedDocument:
    final_url: str
    content_type: str
    body: bytes


class ContentExtractor(Protocol):
    def extract(self, document: FetchedDocument) -> str: ...


class TrafilaturaContentExtractor:
    """Extract readable text only from bytes already fetched by AEGIS."""

    def extract(self, document: FetchedDocument) -> str:
        if document.content_type != "text/html":
            return document.body.decode("utf-8", errors="replace")[:MAX_EXTRACTED_TEXT]
        try:
            extract = getattr(import_module("trafilatura"), "extract")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("Trafilatura is not installed") from exc
        extracted = extract(document.body.decode("utf-8", errors="replace"), include_links=False)
        text = extracted if isinstance(extracted, str) else ""
        if not text:
            raise ValueError("research source contained no readable text")
        return text[:MAX_EXTRACTED_TEXT]


def _validated_url(value: str) -> tuple[str, str, int | None, str]:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("research fetch only supports HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("research URL cannot contain credentials")
    if not parsed.hostname:
        raise ValueError("research URL requires a host")
    port = parsed.port
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("research URL port is invalid")
    return (
        parsed.scheme,
        parsed.hostname,
        port,
        urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")),
    )


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.is_global and not ip.is_private and not ip.is_loopback


Resolver = Callable[[str, int], tuple[str, ...]]


def resolve_public(host: str, port: int) -> tuple[str, ...]:
    addresses = tuple(
        sorted(
            {
                cast(str, item[4][0])
                for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            }
        )
    )
    if not addresses or any(not _is_public(address) for address in addresses):
        raise ValueError("research destination is not globally routable")
    return addresses


class DocumentFetcher:
    """Fetch bounded public HTML while connecting to a validated IP address."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        resolver: Resolver = resolve_public,
        max_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        if timeout <= 0 or max_bytes <= 0:
            raise ValueError("research fetch bounds must be positive")
        self.timeout = timeout
        self.resolver = resolver
        self.max_bytes = max_bytes

    def fetch(self, url: str) -> FetchedDocument:
        current = url
        for redirect in range(MAX_REDIRECTS + 1):
            scheme, host, port, normalized = _validated_url(current)
            target_port = port or (443 if scheme == "https" else 80)
            addresses = self.resolver(host, target_port)
            if not addresses or any(not _is_public(address) for address in addresses):
                raise ValueError("research destination is not globally routable")
            response = self._request(scheme, host, target_port, normalized, addresses[0])
            status, headers, body, location = response
            if len(body) > self.max_bytes:
                raise ValueError("research response exceeds size bound")
            if status in {301, 302, 303, 307, 308}:
                if redirect >= MAX_REDIRECTS or not location:
                    raise ValueError("research redirect limit exceeded")
                current = urljoin(normalized, location)
                continue
            if status < 200 or status >= 300:
                raise ValueError("research source returned an unusable response")
            content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type not in {"text/html", "text/plain"}:
                raise ValueError("research source content type is unsupported")
            return FetchedDocument(normalized, content_type, body)
        raise AssertionError("bounded redirect loop unexpectedly continued")

    def _request(
        self, scheme: str, host: str, port: int, url: str, address: str
    ) -> tuple[int, dict[str, str], bytes, str | None]:
        connection: HTTPConnection | HTTPSConnection
        if scheme == "https":
            connection = _PinnedHTTPSConnection(host, address, port, self.timeout)
        else:
            connection = _PinnedHTTPConnection(host, address, port, self.timeout)
        try:
            parsed = urlsplit(url)
            path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            connection.request(
                "GET", path, headers={"Host": parsed.netloc, "Accept": "text/html,text/plain"}
            )
            response = connection.getresponse()
            body = response.read(self.max_bytes + 1)
            if len(body) > self.max_bytes:
                raise ValueError("research response exceeds size bound")
            return (
                response.status,
                {key.lower(): value for key, value in response.getheaders()},
                body,
                response.getheader("Location"),
            )
        finally:
            connection.close()


class _PinnedHTTPConnection(HTTPConnection):
    def __init__(self, host: str, address: str, port: int, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(HTTPSConnection):
    def __init__(self, host: str, address: str, port: int, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout, context=create_default_context())
        self._address = address
        self._tls_context = create_default_context()
        self._server_hostname = host

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._tls_context.wrap_socket(sock, server_hostname=self._server_hostname)


class SearxngSearchProvider:
    """SearXNG's documented operator-configured HTTP/JSON endpoint."""

    provider_id = "searxng"

    def __init__(self, endpoint: str, timeout: float = 10.0) -> None:
        scheme, _host, _port, normalized = _validated_url(endpoint)
        if scheme not in {"http", "https"}:
            raise ValueError("SearXNG endpoint must use HTTP(S)")
        self.endpoint = normalized.rstrip("/")
        self.timeout = timeout

    def search(self, request: SearchRequest) -> tuple[SearchCandidate, ...]:
        query = urlencode(
            {"q": request.query, "format": "json", "number_of_results": request.limit}
        )
        try:
            with urlopen(f"{self.endpoint}/search?{query}", timeout=self.timeout) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("SearXNG search failed") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RuntimeError("SearXNG returned no usable JSON results")
        candidates: list[SearchCandidate] = []
        for item in payload["results"][: request.limit]:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            candidates.append(
                SearchCandidate(
                    title=str(item.get("title", "Untitled source")),
                    url=item["url"],
                    snippet=str(item["content"]) if isinstance(item.get("content"), str) else None,
                )
            )
        return tuple(candidates)


class ResearchUnavailable(RuntimeError):
    """No bounded external evidence could be obtained."""


class ResearchService:
    """Collect bounded public evidence for an answer-only caller."""

    def __init__(
        self, provider: SearchProvider, fetcher: DocumentFetcher, extractor: ContentExtractor
    ) -> None:
        self.provider = provider
        self.fetcher = fetcher
        self.extractor = extractor

    def collect(self, request: SearchRequest) -> EvidenceSet:
        candidates = self.provider.search(request)
        seen: set[str] = set()
        evidence: list[Evidence] = []
        for candidate in candidates:
            try:
                _scheme, _host, _port, normalized = _validated_url(candidate.url)
                if normalized in seen:
                    continue
                seen.add(normalized)
                document = self.fetcher.fetch(normalized)
                text = self.extractor.extract(document)
                evidence.append(
                    Evidence(
                        source_id=f"source-{len(evidence) + 1}",
                        final_url=document.final_url,
                        title=candidate.title[:500] or "Untitled source",
                        text=text,
                        retrieved_at=datetime.now().astimezone(),
                        published_at=candidate.published_at,
                        snippet=candidate.snippet,
                    )
                )
            except (ValueError, RuntimeError, OSError):
                continue
            if len(evidence) >= request.limit:
                break
        if not evidence:
            raise ResearchUnavailable("no usable external evidence was obtained")
        return EvidenceSet(
            query=request.query,
            provider_id=self.provider.provider_id,
            evidence=tuple(evidence),
            retrieved_at=datetime.now().astimezone(),
        )

    def answer(
        self,
        question: str,
        request: SearchRequest,
        synthesizer: AnswerSynthesizer,
        local_context: dict[str, object] | None = None,
    ) -> ResearchAnswer:
        """Synthesize only from fetched evidence; this path cannot execute actions."""

        evidence = self.collect(request)
        text = synthesizer.synthesize(question, evidence, local_context)
        source_kind = KnowledgeSource.MIXED if local_context else KnowledgeSource.EXTERNAL
        return ResearchAnswer(text=text, source_kind=source_kind, evidence=evidence)
