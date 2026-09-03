# Structural fidelity Qwen3:8B guarded probe

This is a development-only probe, not a held-out promotion evaluation.

- Source: `dc94e0b`
- Model: `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Corpus: five fresh compound cases; SHA-256 `7720b8b95df210d5983b6c0268609371e6fd96c53b864e5cd44609914e89fd5d`
- Structural model: `en_core_web_sm` 3.8.0 with spaCy 3.8.16
- Model calls: 7 total, 1.4/request; decoder failures: 0
- Mean latency: 4487 ms; request P50: 4120 ms; request P95: 6337 ms
- False mutations: **0**; false completions: **0**; unsafe mutations/1000: **0**
- Compound route accuracy: 0.8; semantic-mode accuracy: 0.8
- Inappropriate clarification rate: 1.0 on this intentionally compound-heavy probe

The prior unguarded run on the same corpus produced two false mutations in the
negation/correction cases. The structural write gate removed both mutations by
clarifying before execution. The result therefore establishes the safety gate,
not owner capability-green performance: Qwen3:8B still over-clarifies and needs
future capability/model work outside this bounded parser campaign.
