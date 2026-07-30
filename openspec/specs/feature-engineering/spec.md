## Purpose

Turn stored Formula 1 results into a modelling panel of one row per driver per round,
where every value describes only what was knowable before that round was run.

## Requirements

### Requirement: No feature may depend on the future

Every engineered value at a given round SHALL be computable from data available strictly
before that round's result. Rolling aggregations MUST apply a one-round shift before
aggregating, so a row describes how a driver arrived at a race rather than how it finished.

The automated guard for this covers the shape of the shift, on a fixture where every driver
contests every round at a distinct circuit. It does not exercise absences, debutants or
repeated circuits, so it cannot detect a join defect or a skewed denominator — see the
audit's D5. Treat a passing guard as necessary, not sufficient.

#### Scenario: Deleting the future leaves the past unchanged

- **WHEN** the pipeline runs over a full history, and again over the same history truncated
  at a chosen round
- **THEN** every row visible at that cutoff has byte-for-byte identical feature values in
  both runs

### Requirement: Feature blocks declare and check their inputs

Each block SHALL declare the columns it requires, and the pipeline MUST verify those
columns exist before running the block, so a misordered or disabled block fails loudly
rather than producing silently wrong values.

#### Scenario: A block with unmet requirements does not run

- **WHEN** a block declares a column that no earlier block produced
- **THEN** the pipeline raises rather than emitting the block's columns as null

### Requirement: Rounds a driver did not contest are absent, not zero

Where a driver missed a round, form features SHALL be computed over that driver's own
recent appearances rather than treating the missed round as a result. A driver who did not
race MUST NOT be recorded as having performed badly.

#### Scenario: A skipped round does not distort a trend

- **WHEN** a driver's last three appearances span four calendar rounds
- **THEN** the fitted trend uses the three appearances, in order, with no gap term

### Requirement: Engineered values keep their numeric type

Continuous features SHALL be produced as floating-point values. A continuous quantity MUST
NOT be silently narrowed to an integer, which would discard the precision the feature
exists to carry.

#### Scenario: A fitted slope keeps its fractional part

- **WHEN** the slope over a driver's recent finishes is 1.5
- **THEN** the feature holds 1.5, not 1
