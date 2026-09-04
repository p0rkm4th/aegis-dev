# Qwen3:8B validation-guided recovery probe

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `566fdfe`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Hardware: Hades RTX 3080 Ti Laptop, 15.6 GiB VRAM reported by Ollama
- Execution: disabled; evaluation boundary only

## Result

The run made 105 model calls, used 252,193 prompt tokens and 11,253 output tokens, and recorded 19 bounded repair attempts across 18 requests. P50/P95 full-request latency was 10.8s/29.5s.

For the 20 `supported_clear` cases, final action-compatible success was 1/20 (5%), semantic-mode accuracy was 10%, and inappropriate clarification was 17/20 (85%). Recovery did not reach a validated compound plan in this probe. Repair failures were chiefly repeated `UNACCOUNTED_STRUCTURAL_ANCHOR`/`BAD_SOURCE_SPAN`; malformed responses often repaired to `CLARIFY`.

Safety invariants held: Core false completions 0, false mutations 0, unsafe mutations per 1000 0, and no security hard failure. This is evidence that the recovery contract contains model errors, but not yet evidence of product-useful compound recovery. The next experiment should address the effect-representation usability/correlation boundary without weakening structural, objective, authorization, Kernel, observation, or verification gates.
