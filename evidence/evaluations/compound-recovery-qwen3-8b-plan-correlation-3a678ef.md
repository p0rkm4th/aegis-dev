# Qwen3:8B repaired plan/effect correlation replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases; unchanged)
- Source revision: `3a678eff1f32aee716d2069f4c6ef107117c4b96`
- Provider: Ollama Qwen3:8B, unchanged digest `500a1f067a9f...b2b8b41`
- Independent structural model: `en_core_web_sm`
- Report: `evaluation/reports/qwen3-8b-compound-41-plan-correlation-3a678ef.json`

Core now checks repaired plan steps against the independently grounded
objective requirements before downstream plan materialization. The replay made
123 model calls, 17 repair requests, and 30 bounded attempts. Supported-clear
final correctness remained 2/20 (10%) with 90% inappropriate clarification;
overall route accuracy was 46.34% and overall correct completion/answer rate
was 15.38%, identical to the preceding repair-contract replay.

Safety remained green: false completions 0, false mutations 0, unsafe mutations
0, and security hard failure false. Hosted validation run 33911232202 passed.
This closes a correlation containment gap but produces no capability-green claim;
owner promotion is not warranted.
