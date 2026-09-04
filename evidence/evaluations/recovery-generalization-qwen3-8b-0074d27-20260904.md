# Contrast-gated compound repair: Qwen3:8B

Source `0074d27` prevented negated or contrastive structural evidence from
entering the new pre-fidelity compound repair path, while preserving bounded
repair for ordinary compound proposals. On the unchanged 20-case
generalization corpus, final correctness rose to 30%, supported-clear success
was 2/8, and inappropriate clarification was 50%. The run used 74 model calls,
12 repair requests, and 20 bounded attempts.

The correction improved the prior tradeoff but did not make the evaluator
safety-green: two false-completion and three false-mutation proposals remained.
No Kernel execution occurred. The source is therefore not eligible for owner
promotion; the owner runtime remains on the previously proven `987b779` release.
The next hypothesis must address referent/argument grounding for the remaining
unsafe proposals without weakening deterministic validation.
