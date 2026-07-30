## Purpose

Report which inputs drove a model's prediction for a given round, so a served probability
can be interrogated rather than taken on trust.

## Requirements

### Requirement: Attribution explains the uncalibrated model

Attribution SHALL be computed against the fitted estimator before probability calibration,
since calibration rescales outputs without changing which inputs the model responded to.

#### Scenario: A model with no logged pre-calibration pipeline is refused

- **WHEN** attribution is requested for a model version that logged no pre-calibration
  pipeline
- **THEN** the request fails with that reason rather than explaining the calibrated wrapper

### Requirement: The explainer matches the algorithm

The system SHALL dispatch to an algorithm-native explainer where one exists, and fall back
to a sampled model-agnostic explainer otherwise. The explainer MUST see exactly what the
estimator sees, with any preprocessing already applied.

#### Scenario: Preprocessing is applied before explaining

- **WHEN** a pipeline includes an imputation or scaling step
- **THEN** attribution is computed on the transformed matrix, not the raw inputs

### Requirement: Attribution is deterministic and reusable

Given the same data, model version and round, attribution SHALL return the same result, and
repeated requests MUST NOT recompute it. Reuse MUST be invalidated by a change to the
source data.

#### Scenario: Revisiting an explanation is immediate

- **WHEN** the same round and model version are explained twice
- **THEN** the second request returns the stored result without recomputing
