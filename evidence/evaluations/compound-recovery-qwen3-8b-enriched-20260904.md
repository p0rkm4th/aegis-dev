# Qwen3:8B enriched structural-repair replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `a91ab9f`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

The replay made 105 model calls and 19 bounded repair attempts. P50/P95 full-request latency was 11.0s/30.6s. Supported-clear inappropriate clarification remained 17/20 (85%) and final compound completion remained 0/20. Repair outcomes remained concentrated in repeated `UNACCOUNTED_STRUCTURAL_ANCHOR`, `BAD_SOURCE_SPAN`, or decoded `CLARIFY` results despite the new bounded count/span diagnostics.

Safety remained green: zero Core false completions, false mutations, unsafe mutations, and security hard failures. The evidence discriminators are useful telemetry but did not make the model repair contract effective; the next hypothesis must address effect representation/segmentation itself, not merely add more failure text or route around validators.
