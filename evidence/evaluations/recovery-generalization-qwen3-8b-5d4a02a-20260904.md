# Compound-repair handoff experiment: Qwen3:8B

Source `5d4a02a` moved incomplete single-action proposals through bounded PLAN
repair before objective-fidelity validation. On the unchanged 20-case
generalization corpus, final correctness improved to 20% and inappropriate
clarification fell to 50%. Repair activity increased to 14 requests and 24
bounded attempts.

The tradeoff failed the safety gate. The execution-free evaluator flagged two
false completions and five false mutations, including new proposals on
negation/correction cases. No Kernel execution occurred, but the source is not
promotion eligible. The next action is to preserve the usability signal while
adding a stricter typed validation condition or rolling back the repair path;
no authority or safety validator may be weakened.
