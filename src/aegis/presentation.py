"""Safe owner-facing rendering helpers for bounded assistant content."""

from __future__ import annotations

import re
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit

from markdown import markdown

_ALLOWED_TAGS = {
    "a",
    "blockquote",
    "br",
    "code",
    "del",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}
_VOID_TAGS = {"br"}
_SAFE_SCHEMES = {"http", "https", "mailto"}
_LANGUAGE_CLASS = re.compile(r"language-[A-Za-z0-9_+-]{1,32}\Z")


class _SafeHTML(HTMLParser):
    """Keep the small semantic HTML subset produced by Python-Markdown."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(escape(data, quote=False))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _ALLOWED_TAGS:
            return
        safe_attrs: list[str] = []
        for name, value in attrs:
            if value is None:
                continue
            if tag == "a" and name == "href" and _safe_url(value):
                safe_attrs.append(f' href="{escape(value, quote=True)}"')
            elif tag == "a" and name == "title":
                safe_attrs.append(f' title="{escape(value, quote=True)}"')
            elif tag == "code" and name == "class" and _LANGUAGE_CLASS.fullmatch(value):
                safe_attrs.append(f' class="{escape(value, quote=True)}"')
        if tag == "a" and any(attr.startswith(" href=") for attr in safe_attrs):
            safe_attrs.extend([' target="_blank"', ' rel="noopener noreferrer"'])
        self.parts.append(f"<{tag}{''.join(safe_attrs)}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _ALLOWED_TAGS and tag not in _VOID_TAGS:
            self.parts.append(f"</{tag}>")


def _safe_url(value: str) -> bool:
    parsed = urlsplit(value)
    return not parsed.scheme or parsed.scheme.lower() in _SAFE_SCHEMES


def render_safe_markdown(text: str) -> str:
    """Render common Markdown while treating model output as untrusted content."""

    escaped_source = escape(text, quote=False)
    rendered = markdown(escaped_source, extensions=["fenced_code", "tables", "sane_lists"])
    sanitizer = _SafeHTML()
    sanitizer.feed(rendered)
    sanitizer.close()
    return "".join(sanitizer.parts)
