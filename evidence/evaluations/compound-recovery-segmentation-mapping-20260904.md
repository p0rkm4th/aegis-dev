# Segmentation/mapping contract replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `381e8f9`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

The first replay after separating unbound effect segmentation from capability mapping made 105 model calls and 19 bounded repairs. Supported-clear inappropriate clarification remained 17/20 (85%), final compound completion remained 0/20, and route accuracy remained 41.5%. Safety stayed green: zero false completions, false mutations, unsafe mutations, and security hard failures.

The run did not expose enough provider-mode telemetry to prove how often the model emitted `action_ref=null`, so no claim is made that the new mapping pass was exercised. The next checkpoint is provider-boundary instrumentation for request-mode counts, followed by a focused replay; no additional authority is granted.
