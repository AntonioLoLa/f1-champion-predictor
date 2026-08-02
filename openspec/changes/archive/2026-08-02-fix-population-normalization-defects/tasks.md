## 1. Implementation (already shipped)

- [x] 1.1 Remove `exclude_non_starters` filtering from `dataset/builder.py::_apply_filters`
      and `conf/features.yaml`, so the scored population is every driver still
      contesting the championship, not just those who started a given round.
- [x] 1.2 Zero a mathematically eliminated driver's probability before rescaling, in
      `models/normalizer.py::normalize_round_probabilities`, applied identically in
      training and inference.
- [x] 1.3 Make `group_log_loss` raise rather than silently skip a group that does not
      contain exactly one positive row, so a population defect can no longer hide.

## 2. Verification

- [x] 2.1 Rebuild the dataset and confirm the row count matches the population inference
      always saw (19,195 → 21,072 rows).
- [x] 2.2 Retrain and confirm `group_log_loss` no longer skips any group.
- [x] 2.3 Reproduce the audit's own evidence scenario live (2020 Sakhir GP, round 16,
      Hamilton absent) and confirm the champion is now scored correctly instead of a
      mathematically-eliminated driver receiving the rescaled mass.

## 3. Spec

- [x] 3.1 Write the delta spec strengthening both requirements and replacing the two
      defect-documenting scenarios.
- [x] 3.2 Archive this change into `openspec/specs/championship-prediction/spec.md`.
