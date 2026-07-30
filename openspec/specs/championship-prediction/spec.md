## Purpose

Serve each driver's probability of winning the championship as of a given round, scored
against data as it stands right now.

## Requirements

### Requirement: Predictions reflect currently stored data

Features used to score a prediction SHALL be rebuilt from the database at request time,
not read from a dataset snapshot, so a prediction never lags behind what has been ingested.

#### Scenario: Newly ingested results change the next prediction

- **WHEN** a round's results are ingested and a prediction is requested
- **THEN** the prediction accounts for that round without any rebuild step in between

### Requirement: Probabilities are shares of one championship

Exactly one driver wins a season, so served probabilities SHALL be rescaled to sum to one
within a season and round. Both the pre-rescaling and post-rescaling values MUST be
available, so the rescaling is inspectable rather than hidden.

Rescaling currently assumes the eventual champion is among the drivers scored for the
round. Where they are not, the remaining mass is redistributed over the drivers who are,
and the served probability can be confident and wrong even though the model's
pre-rescaling output was not. This is a known defect, not a guarantee — see the audit's
E4 and the entry it drives in `ROADMAP.md`.

#### Scenario: A round's probabilities sum to one

- **WHEN** a round is scored
- **THEN** the served probabilities across all drivers in that round sum to one

#### Scenario: A missing driver distorts the rescaling

- **WHEN** the driver who went on to win the title is absent from a round's scored
  population
- **THEN** the rescaled probabilities still sum to one over the drivers present, and no
  longer describe that round's real championship odds

### Requirement: The scored population is documented

The set of drivers a round is scored over SHALL be knowable from the response, because it
determines the denominator every rescaled probability depends on.

Training and serving do not currently agree on that set: the dataset builder applies
filters that inference does not, so the live panel carries roughly 9% more rows than the
model was fitted on. Aligning the two is the audit's highest-priority fix (E3).

#### Scenario: Serving and training populations differ

- **WHEN** the same round is scored live and looked up in the training dataset
- **THEN** the live population may contain drivers the training population excluded

### Requirement: A prediction names the model that produced it

Every prediction SHALL identify the model version it was scored with, so results from
different candidates are never confused.

#### Scenario: Scoring with a chosen candidate reports that candidate

- **WHEN** a prediction is requested for a specific model version
- **THEN** the response names that version rather than the serving default

### Requirement: A model is refused rather than misapplied

If the current feature configuration no longer produces a column a model was trained on,
the system SHALL raise and name the missing columns rather than scoring on a different
feature set.

#### Scenario: A retired feature blocks scoring

- **WHEN** a model's training-time features are no longer all produced
- **THEN** scoring fails with the missing column names and a direction to retrain

### Requirement: Repeated work is not repeated

Rebuilding the feature panel SHALL be avoided when the underlying data has not changed,
and any such reuse MUST be invalidated by a change to the source data, including an
in-place correction that leaves row counts unchanged.

#### Scenario: Switching model version does not rebuild features

- **WHEN** the same round is scored again with a different model version
- **THEN** the feature panel is reused and only the scoring step repeats

#### Scenario: Corrected data invalidates the reuse

- **WHEN** an ingestion run rewrites existing rows without changing any row count
- **THEN** the next prediction rebuilds the feature panel
