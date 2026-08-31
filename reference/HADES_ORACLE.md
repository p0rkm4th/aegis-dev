# Frozen Hades oracle

The latest pushed `hades-v1-productization` head was frozen at
`65ac52f2f0ba7d468644d4ff84460f00cb7e1cd8` on 2026-08-31. Hades feature
development is over. Aegis may use the committed artifacts as behavioral,
evaluation, and reproducibility references only. Its AGPL implementation and
tests are not copied into Aegis.

Initial oracle cases to reconstruct independently are deterministic reads,
structured ACTION/ANSWER/NEED_CONTEXT/CLARIFY/BLOCKED decisions, malformed or
invented actions, policy denial, approval, execution failure, verification
failure, continuation, replay/recovery, and evidence-grounded completion.
