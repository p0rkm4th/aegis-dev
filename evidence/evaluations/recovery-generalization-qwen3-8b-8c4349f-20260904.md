# Structural repair evidence experiment: Qwen3:8B

Source `8c4349f` exposed bounded structural anchor counts and source spans to
clarification-triggered repair proposals. The unchanged 20-case corpus was
replayed with the same model digest and contracts.

The hypothesis did not produce a measurable capability improvement: final
correctness was 10%, route accuracy 35%, semantic-mode accuracy 30%, and
inappropriate clarification 70%. There were 53 model calls, 6 repair requests,
and 7 bounded attempts. The evaluator again flagged two false-completion and
three false-mutation proposals; no Kernel execution occurred.

This is a failed experiment, not promotion evidence. The bounded span data is
useful diagnostic telemetry, but it did not resolve the dominant model behavior.
The next repair hypothesis must address the structural/decoder contract more
directly; safety validators and authority boundaries remain unchanged.
