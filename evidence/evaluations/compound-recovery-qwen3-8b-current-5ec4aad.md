# Qwen3:8B frozen compound recovery replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases; unchanged)
- Source revision: `5ec4aada397d5646dd00bd8fbdf09963707ab17b`
- Provider: Ollama Qwen3:8B, endpoint `http://172.18.0.1:11434`
- Independent structural model: `en_core_web_sm`
- Report: `evaluation/reports/qwen3-8b-compound-41-current-5ec4aad.json`

The replay made 118 model calls, including 17 repair requests and 27 bounded
repair attempts. Supported-clear final valid-route acceptance was 1/20 (5%),
with 85% inappropriate clarification. Overall valid-route rate was 11.54%.
Full-request p50/p95 latency was 10.91s/27.35s.

Safety remained green: Core false completions 0, false mutations 0, unsafe
mutations 0, and security hard failure false. Decoder failures were 2.

This confirms the resident model's compound cognition remains below the 80%
product target; validators and authority were not weakened. The next model
hypothesis must be justified by the failure telemetry rather than benchmark-
specific prompt tuning.
