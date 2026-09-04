# Qwen3:14B current-contract recovery replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases; unchanged)
- Source revision: `646efd1f99adda9923478900c5c44709fcc8f5cd`
- Provider: Ollama Qwen3:14B, digest `bdbd181c...9debe8`
- Independent structural model: `en_core_web_sm`
- Report: `evaluation/reports/qwen3-14b-compound-41-current-646efd1.json`

This was an execution-free same-contract hard-cognition control replay. It made
132 model calls across 19 repair requests and 24 bounded repair attempts; the
maximum remained two repair attempts per case. Supported-clear final correctness
was 4/20 (20%), route and semantic-mode accuracy were 39.02% overall, and
inappropriate clarification was 76.92% overall (75% for supported-clear).
Median and p95 full-request latency were 25.45s and 48.21s.

Safety remained green: zero false completions, zero false mutations, zero
unsafe mutations, zero decoder failures, and no security hard failure. Qwen3:14B
is retained as a measured diagnostic/hard-cognition reference, not promoted to
the resident path and not used to justify automatic routing.
