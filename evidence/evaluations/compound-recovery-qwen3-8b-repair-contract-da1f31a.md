# Qwen3:8B structural repair-contract replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases; unchanged)
- Source revision: `da1f31a5282670177eb9cc8aca5b7a8b82de252d`
- Provider: Ollama Qwen3:8B, unchanged digest `500a1f067a9f...b2b8b41`
- Independent structural model: `en_core_web_sm`
- Report: `evaluation/reports/qwen3-8b-compound-41-repair-contract-da1f31a.json`

The provider-boundary contract for `UNACCOUNTED_STRUCTURAL_ANCHOR` repairs now
requires an attempted complete PLAN rather than permitting avoidable CLARIFY.
The replay made 123 model calls, 17 repair requests, and 30 bounded attempts.
Repair telemetry showed PLAN outputs on several previously immediate-clarify
cases, but supported-clear final correctness remained 2/20 (10%) with 90%
inappropriate clarification. Overall route accuracy was 46.34% and overall
correct completion/answer rate was 15.38%.

Safety remained green: Core false completions 0, false mutations 0, unsafe
mutations 0, and security hard failure false. This is a useful contract
improvement but not capability-green; no owner promotion is justified.
