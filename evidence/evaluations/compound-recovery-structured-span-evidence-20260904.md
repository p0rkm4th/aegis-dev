# Structured span evidence replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `619890a57396b507ebc7ab29138212ab77a883b2`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

Core now carries bounded unmatched source positions as structured failure evidence in addition to the privacy-minimal detail string. The same-contract replay made 117 model calls across 18 repair requests and 28 repair attempts, with maximum per-case attempts 2. Modes were `effect_segmentation=7`, `effect_repair=5`, `objective_mapping=2`, and `ordinary_decision=103`.

Capability did not improve: overall valid-route was 7.69%, supported-clear valid-route 1/20 (5%), and inappropriate clarification 85%. Failure kinds remained `UNACCOUNTED_STRUCTURAL_ANCHOR=13`, `DECODER_SCHEMA_FAILURE=12`, `BAD_SOURCE_SPAN=2`, and `CANONICAL_CONTRADICTION=1`.

Safety remained green: zero false completions, false mutations, unsafe mutations, and security hard failures. The structured evidence is retained because it improves contract fidelity and deterministic observability, but this is a failed capability experiment, not a management stop.
