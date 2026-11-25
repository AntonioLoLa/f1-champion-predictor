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

The scored population for a round SHALL be every driver still contesting that
championship, whether or not they started that particular round, so the eventual champion
is never absent from the group being rescaled. A driver who is mathematically eliminated
SHALL have their probability set to zero before the rescale, so eliminated drivers cannot
absorb probability mass that belongs to drivers who can still win.

#### Scenario: A round's probabilities sum to one

- **WHEN** a round is scored
- **THEN** the served probabilities across all drivers in that round sum to one

#### Scenario: A driver who misses a round is still scored

- **WHEN** a driver contesting the championship does not start a given round
- **THEN** that driver still has a row in the scored population for that round, so the
  eventual champion is never dropped from the group being rescaled

#### Scenario: An eliminated driver never receives rescaled probability mass

- **WHEN** a driver is mathematically eliminated from the championship as of a round
- **THEN** that driver's rescaled probability for that round is zero

### Requirement: The scored population is documented

The set of drivers a round is scored over SHALL be knowable from the response, because it
determines the denominator every rescaled probability depends on. Training and serving
SHALL agree on that set, so a live prediction scores exactly the population the model was
evaluated against.

#### Scenario: Serving and training populations agree

- **WHEN** the same round is scored live and looked up in the training dataset
- **THEN** the two populations are identical

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

### Requirement: A round's ranking is deterministic

Drivers in a round SHALL be ranked by their rescaled probability, highest first. Drivers
with exactly equal rescaled probabilities SHALL be ordered by their championship standing
as of that round: a better position first, a driver with no position after every driver
with one, and the driver's name as a last resort. Unequal probabilities MUST keep the
model's order, however small the difference.

#### Scenario: A decided title ranks the field by the standings

- **WHEN** a round is scored after which every driver but the leader is mathematically
  eliminated
- **THEN** the leader is ranked first and every other driver follows in championship
  position order

#### Scenario: The model's order survives a near tie

- **WHEN** two drivers have different rescaled probabilities and the lower one stands
  higher in the championship
- **THEN** the driver with the higher probability is still ranked first

### Requirement: A prediction identifies each driver in the reader's terms

Each driver in a served prediction SHALL carry the team's display name alongside its
identifier, and the driver's championship points and position as of that round, so a
consumer can present and order the field without a second request.

#### Scenario: A prediction names the team

- **WHEN** a round is scored for a driver whose team is in the constructors table
- **THEN** the driver's row carries that team's name as well as its identifier
