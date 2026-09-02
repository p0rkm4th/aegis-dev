# Open-source harvest record

This is a dated research record, not a dependency lock. Upstream revisions are
recorded so later adoption can be deliberate and license/provenance reviewed.
No upstream code is currently imported by AEGIS as a result of this audit.

| Candidate and path | License / revision inspected | Classification | AEGIS capability | Integration and ownership decision |
| --- | --- | --- | --- | --- |
| [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), `agent_loop.py`, `tests/` | MIT, `9514d354ca47` | TEST-HARVEST / DESIGN-HARVEST | Multi-turn agent evaluation vocabulary, trial/outcome grading, agent-loop test patterns | Reuse concepts only. Its agent loop and runtime assumptions would duplicate AEGIS Core ownership; no code deletion. |
| [openclaw/openclaw](https://github.com/openclaw/openclaw), gateway protocol/client paths | `NOASSERTION`, `1160e395e452` | REJECT pending license/provenance clarification | Gateway protocol, reconnect, device auth, channel transport | Do not vendor or depend on this revision. AEGIS keeps its narrow typed transport and may consume an explicitly licensed schema/client later; authority remains in Core. |
| [cytoscape/cytoscape.js](https://github.com/cytoscape/cytoscape.js), `documentation/` and graph APIs | MIT, `fd3595bbf0ea` | DESIGN-HARVEST | Constellation graph interaction, layouts, semantic detail projections | Candidate for a later standalone client. Frontend must consume authorized canonical projections; no current dependency or custom graph deletion. |
| [xyflow/xyflow](https://github.com/xyflow/xyflow), React Flow packages/examples | MIT, `0a1f9575b256` | DESIGN-HARVEST | Accessible node/edge browser shell and conventional detail panels | Candidate only if a React client is selected. It cannot own AEGIS state, policy, or graph ontology; no current dependency. |
| [openai/simple-evals](https://github.com/openai/simple-evals), `simple_evals/` | MIT, `652c89d0ca9d` | DESIGN-HARVEST | Dataset-driven model evaluation and sampler separation | AEGIS needs local-provider, state-outcome, and authority-specific scoring, so direct adoption would be misleading. Reuse split/runner ideas; no code deletion. |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), `lm_eval/` and task validation | MIT, `b954108c9baa` | DESIGN-HARVEST | Versioned task corpus validation and reproducible evaluation runs | Broad academic benchmark machinery is heavier than the interaction-boundary harness. Harvest task metadata/validation patterns; do not add it to runtime or CI. |
| [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy), evaluation metrics docs | MIT, `59ce7601ec40` | DESIGN-HARVEST / REJECT FOR RUNTIME | Metric contracts and per-trial result reporting | LLM-judge/optimizer layers would add inference hops and blur AEGIS authority. Keep deterministic outcome graders and provider-neutral contracts; no dependency. |

## Current conclusion

The smallest safe choice is to retain AEGIS's existing evaluation boundary and
extend its corpus/metrics rather than import a framework. Graph libraries are
future client candidates, not Core dependencies. Gateway reuse remains gated
on a clear compatible license and a measured language-boundary benefit. This
record does not claim that any upstream implementation was security-audited by
AEGIS.
