# Structural fidelity owner-usability campaign

## Result

**ESCALATE** the remaining owner-usability/model limitation. The optional
spaCy adapter is retained as the current structural provider (`KEEP / ADAPT`
at the provider boundary), but the campaign's owner capability threshold is
not green: Qwen3:8B over-clarifies the compound-heavy supported sample.

## Source and hosted truth

- Source: `2f07c4d3583f9bc6930ccf9292e64e9e8cfc50db`
- Hosted validation: GitHub Actions `33820495721`, success
- Installed/running/live-green owner release: `8d61b05`
- Owner service health: PostgreSQL, Qwen3:8B, identity, and structural parser
  available; no owner database reset was performed.

## Provider and resources

- Adapter: `SpacyStructuralParser`
- spaCy: `3.8.16`, MIT
- model: `en_core_web_sm 3.8.0`, CC BY 4.0
- Provisioning: optional `structural` extra with exact spaCy pin, plus the
  separately sourced exact model wheel; model is not vendored.
- Owner measurements: approximately 605 ms cold load, 2.29 ms warm p50,
  2.39 ms warm p95 over 300 parses, and approximately 158 MB peak RSS for
  the parser process. The parser is loaded once per process.

## Safety and product evidence

- Guarded Qwen3:8B probe: 5 fresh compound cases, 7 model calls, 0 decoder
  failures, 0 false mutations, 0 false completions, and 0 unsafe mutations.
- The corresponding unguarded probe had 2 false mutations in negation /
  correction cases; the structural write gate caught both.
- Guarded compound route/semantic-mode accuracy: `0.8` / `0.8`.
- Inappropriate clarification on the intentionally compound-heavy probe:
  `1.0`; this fails the owner usability target of at least 80% safe acceptance
  and at most 20% false clarification for supported-clear compounds.
- Installed owner session: 20/20 requests completed without transport failure;
  parser availability and safety were verified, but compound capability was
  not claimed green.
- Core false acceptances, false completions, unsafe executed mutations, and
  blind replays: zero in the guarded evidence and owner checkpoint.

## Diagnosis and next decision

Parser performance and structural safety are acceptable. The dominant
remaining limitation is Qwen/requested-effect interpretation and its
interaction with conservative structural coverage, not evidence that a
second parser or a weaker safety gate is needed. Future work should adapt
compound effect interpretation using fresh evidence; this campaign does not
add a custom English parser, a second reviewer model, or a new authority path.

The campaign is closed at this evidence boundary. Capability acquisition,
Pack SDK work, research broadening, and parser-polish work remain out of scope.
