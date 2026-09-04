# Qwen3:8B second structural repair attempt

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases; unchanged)
- Source revision: `0606245a4131e7843d8ea6bfa96eb0b1cfff416b`
- Provider: Ollama Qwen3:8B, unchanged digest `500a1f067a9f...b2b8b41`
- Independent structural model: `en_core_web_sm`
- Report: `evaluation/reports/qwen3-8b-compound-41-second-structural-attempt-0606245.json`

Allowing two structural-plurality repair attempts produced a modest gain:
supported-clear final correctness rose to 3/20 (15%) from 2/20 (10%), and
inappropriate clarification fell to 85% from 90%. Overall correctness remained
15.38% and route accuracy remained 46.34%. The replay made 129 model calls,
18 repair requests, and 34 attempts; no case exceeded two repair attempts.

Safety remained green: zero false completions, false mutations, unsafe
mutations, and security hard failures. This is retained as bounded recovery
evidence, not capability-green evidence.
