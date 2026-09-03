# Objective-fidelity structural signal benchmark

## Scope

This is a bounded prerequisite for action-agnostic `RequestedEffect` evidence.
Structural analysis supplies span/relationship evidence only; it does not
interpret objectives, map capabilities, authorize actions, or determine
completion.

## OSS survey

| Project | Revision inspected | License | Classification | Result |
| --- | --- | --- | --- | --- |
| spaCy | 3.8.16 package metadata; `en_core_web_sm` 3.8.0 probe model | MIT engine; model has separate CC BY 4.0 distribution/licensing | ADAPT (optional boundary) | Dependency parsing exposed coordinated objects/predicates and negation/correction anchors. The dependency/model footprint is too large for the minimum Core install, so only a replaceable adapter is retained. |
| Stanza | 1.10.1 package metadata | Apache-2.0; model/data terms require separate review | DESIGN-HARVEST / REJECT for the current boundary | Useful UD/dependency-parser design, but it requires a model download and adds a second large runtime stack. |
| UDPipe | 1.4.0.1 package metadata | MPL-2.0 engine; each selected model must be licensed separately | DESIGN-HARVEST / REJECT for the current boundary | Small engine is attractive, but no approved model is present in the repository or owner runtime; model licensing cannot be assumed from the engine license. |

The local owner/runtime environment is Python 3.14. None of these engines or
models is installed there. The isolated spaCy probe loaded in about 0.18 s,
parsed the five benchmark families in about 1.8–3.4 ms per sentence, and
occupied about 302 MB with dependencies plus a 12.8 MB model. This is useful
evidence, but not a reason to impose that footprint on the minimum Core install.

## Decision

Do not vendor or silently install a parser in this change. The Core seam is a
parser-neutral `StructuralCoverageSignal`; `SpacyStructuralParser` is an
optional replaceable adapter and its model remains operator-supplied. The
deterministic proof uses controlled structural signals. Production integration
must provide an independently generated signal; absence of that signal is not
evidence of complete coverage. No custom English parser was added.

Reproducible runtime provisioning uses the optional `structural` extra
(`spacy==3.8.16`) and the separately downloaded exact `en_core_web_sm==3.8.0`
wheel. The model is not vendored; operators must verify its source and hash
before installation and set `AEGIS_STRUCTURAL_MODEL` to its installed model
name or path.

The acceptance rule is conservative: the supplied structural outcome anchors
must correspond one-to-one with grounded effect spans. A whole-utterance span
or duplicated span cannot claim several anchors. Unsupported or unmapped
effects remain representable before ActionCard resolution.
