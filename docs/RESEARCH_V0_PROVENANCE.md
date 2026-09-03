# Research V0 provenance

Research V0 uses small replaceable boundaries. Search results and fetched text
are untrusted, non-authoritative evidence and never provide action capability.

- **SearXNG — ADAPT:** documented HTTP/JSON `/search` interface only. The
  endpoint is operator configuration; AEGIS does not vendor SearXNG or accept
  a per-request endpoint. JSON-disabled responses are provider failures.
- **Trafilatura 2.2.0 — DIRECTLY USE:** Apache-2.0 extraction library, pinned
  as an optional `research` dependency. It receives only HTML already fetched
  and bounded by AEGIS; its network-fetch helpers are not used.
- **Crawl4AI — REJECT / DESIGN-HARVEST:** V0 does not add browser automation,
  JavaScript execution, Chromium, crawling queues, or recursive research.

AEGIS retains URL scheme/DNS/redirect/size/timeout policy, source provenance,
deduplication, and all Core authority boundaries.
