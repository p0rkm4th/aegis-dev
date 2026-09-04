# Small challenger recovery control: Qwen2.5:3B

Qwen2.5:3B digest
`357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` was run
against the same 20-case corpus, candidate retrieval, decoder, structural
parser, and recovery contract as Qwen3:8B.

The challenger is not viable for this recovery role: final correctness was 0%,
route and semantic-mode accuracy 10%, and decoder failures numbered 14. It was
faster (p50/p95 5.54s/9.09s) but produced one false completion and two false
mutation proposals. No Kernel execution occurred. The model is rejected for
promotion or routing; Qwen3:8B remains the control while Core-boundary grounding
and evaluation integrity are investigated.
