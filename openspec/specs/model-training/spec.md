## Purpose

Fit and select a champion-prediction model over the sport's history, under a validation
scheme that reflects how the model will actually be used: predicting a season it has never
seen, from seasons that preceded it.

## Requirements

### Requirement: Validation walks forward through seasons

Candidates SHALL be validated by training on a window of past seasons and testing on the
next one. A random split MUST NOT be used, because it would let a season's outcome inform
predictions about that season's own earlier rounds.

#### Scenario: A fold never tests on a season it trained on

- **WHEN** any validation fold is constructed
- **THEN** every test season is strictly later than every training season in that fold

### Requirement: Calibration is fitted on held-out seasons

The probability calibrator SHALL be fitted on seasons withheld from the candidate's own
fit, and MUST NOT refit the underlying estimator.

#### Scenario: Calibration does not alter the model

- **WHEN** a fitted candidate is calibrated
- **THEN** the estimator's own predictions are unchanged and only the probability mapping
  differs

### Requirement: The calibrator suits the number of independent outcomes

Calibration SHALL account for the fact that a season contributes one championship outcome
regardless of how many driver-round rows it produces. A calibrator flexible enough to
interpolate those rows as independent observations MUST NOT be used, since it collapses to
reporting certainty.

#### Scenario: A served probability is never a bare certainty

- **WHEN** any candidate scores a round with a third of the season still to run
- **THEN** no driver is reported at a probability of exactly one

### Requirement: Every candidate is retained, and one is selected

A training run SHALL record every candidate it fitted with its metrics, and mark exactly
one as the run's best on the configured selection metric. Candidates from earlier runs MUST
NOT be overwritten, so the record of what was tried survives.

#### Scenario: A later run does not erase an earlier one

- **WHEN** a second training run completes
- **THEN** the first run's candidates and metrics remain queryable

### Requirement: The serving model is chosen separately from the best model

The model that serves predictions SHALL be recorded independently of the model a run
selected as best, so a human can deliberately serve a different candidate.

#### Scenario: Activating a candidate does not rewrite history

- **WHEN** a candidate other than the run's best is activated
- **THEN** it becomes the serving model while the run's best marking is unchanged
