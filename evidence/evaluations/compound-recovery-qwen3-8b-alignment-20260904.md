# Qwen3:8B structural span-alignment replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `02b055d`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

The same-contract replay after unmatched-anchor/effect diagnostics made 105 model calls and 19 bounded repairs. Supported-clear inappropriate clarification remained 17/20 (85%), final compound completion remained 0/20, and route accuracy remained 41.5%. P50/P95 latency was 11.2s/30.7s. Repair outcomes were unchanged: repeated `UNACCOUNTED_STRUCTURAL_ANCHOR`/`BAD_SOURCE_SPAN` or `CLARIFY`.

Safety remained green: zero false completions, false mutations, unsafe mutations, and security hard failures. The diagnostic refinement is therefore rejected as a capability improvement. The next hypothesis is a minimal effect-contract representation change, with existing structural and objective validators retained.
