## Why

`openspec/specs/championship-prediction/spec.md` currently documents two defects (E3 and
E4 in `AUDIT.md`, item 1 in `ROADMAP.md`) as known, current behaviour rather than as
guarantees the system violates: rescaling can silently distort a round's probabilities
when the eventual champion is missing from the scored population, and the population
inference scores disagreed with the one training was fitted on. Both are now fixed in
code (`dataset/builder.py`, `models/normalizer.py`) and verified live against the exact
scenario the audit used as evidence. The spec needs to catch up so it stops describing a
bug that no longer exists as if it were still true.

## What Changes

- Strengthen "Probabilities are shares of one championship" from a requirement with a
  documented exception to an unconditional guarantee: a driver absent from a round (but
  still contesting the championship) no longer distorts the rescale, because the scored
  population is now every driver still in the championship, and a mathematically
  eliminated driver's probability is zeroed before the rescale rather than left to
  compete for mass with genuine contenders.
- Strengthen "The scored population is documented" so that serving and training
  agreeing on the scored population is the guarantee, not merely made visible.
- Remove the two scenarios that documented the defects ("A missing driver distorts the
  rescaling", "Serving and training populations differ") and replace them with scenarios
  that assert the fixed behaviour.
- No API, schema, or behavioural change beyond what already shipped — this proposal
  formalizes what the code already does.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `championship-prediction`: the two requirements above go from "documented defect" to
  "guarantee", with their scenarios changed accordingly.

## Impact

- Spec only: `openspec/specs/championship-prediction/spec.md`.
- No code changes — `dataset/builder.py::_apply_filters` and
  `models/normalizer.py::normalize_round_probabilities` already implement this.
