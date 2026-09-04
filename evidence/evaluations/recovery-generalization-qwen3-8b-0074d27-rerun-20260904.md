# Contrast-gated compound repair rerun: Qwen3:8B

The unchanged 20-case corpus was rerun at `0074d27` after the typed
contrast/negation gate. The result reproduced the earlier usability improvement:
30% final correctness, 2/8 supported-clear success, and 50% inappropriate
clarification. It used 74 model calls, 12 repair requests, and 20 bounded
attempts.

The evaluator again flagged exactly two false completions and three false
mutations, all in the same three ambiguous completion/scheduling cases. No
Kernel execution occurred. This repeat confirms a stable usability signal but
does not satisfy the zero-unsafe-proposal release bar; the owner runtime was not
promoted. Remaining work is typed referent/argument containment, not prompt
flattery or validator relaxation.
