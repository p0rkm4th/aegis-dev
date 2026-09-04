# Frozen compound recovery control: Qwen3:8B

The unchanged 41-case corpus was replayed at source `1f4de3a` using the same
Qwen3:8B digest and structural model contract. Final correct
completion/answer was 11.54%; supported-clear success was 2/20, route and
semantic-mode accuracy were 43.90%, and inappropriate clarification was
84.62%. The run made 125 model calls, requested 18 repairs, and used 31 bounded
repair attempts. Latency p50/p95 was 11.46s/27.64s.

Safety remained green: zero false completions, false mutations, unsafe
mutations, and security hard failures. Repair failures were concentrated in
`UNACCOUNTED_STRUCTURAL_ANCHOR` (16) and `DECODER_SCHEMA_FAILURE` (9), with
smaller ambiguity/span/mismatch classes. This is control evidence below the
product target, not owner-promotion evidence. The next bounded hypothesis is a
structural repair-contract improvement; validators, authorization, Kernel,
observation, and verification remain unchanged.
