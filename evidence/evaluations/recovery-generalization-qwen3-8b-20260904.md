# Fresh recovery generalization: Qwen3:8B

The frozen `evaluation/recovery_generalization_20.json` corpus contains 20 novel
cases and 32 phenomena with zero normalized overlap against the frozen 41-case
compound corpus. The execution-free evaluator ran against Qwen3:8B digest
`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`.

Results were weak: final correct completion/answer 10%, route accuracy 40%,
semantic-mode accuracy 35%, and inappropriate clarification 70%. There were 51
model calls, 6 repair requests, 8 repair attempts, and 1 decoder failure.

This is a failed safety/generalization experiment, not a promotion signal. The
execution-free evaluator flagged three unsafe proposals: two inferred task
completions and one event creation without a user-supplied time. No Kernel
execution occurred in the evaluator. An owner-runtime probe blocked both
referent completions, but accepted “Schedule the review” before the follow-up
fix; that event was the concrete production defect.

The responsible seam was reference-pack grounding: a model-supplied
`starts_at` was accepted even when the utterance supplied no date or time. The
follow-up fix requires an explicit temporal cue before event creation and adds a
deterministic regression. No safety validator was weakened and the experimental
release is not eligible for owner promotion.
