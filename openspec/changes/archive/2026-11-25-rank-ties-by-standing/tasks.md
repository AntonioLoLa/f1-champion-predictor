## 1. Backend

- [x] 1.1 Break exact ties in the rank by championship position, then name, in
      `inference/predictor.py::_score`.
- [x] 1.2 Add `points`, `standing_position` and `constructor_name` to `DriverPrediction`
      and `DriverPredictionResponse`.
- [x] 1.3 Add `constructor_name` to standings and race results, and
      `winner_constructor_name` to the season calendar.
- [x] 1.4 Add `rounds_remaining` to the title-contention response.
- [x] 1.5 Name each standing's team from the driver's latest race result, not from the
      provider's list of every team they raced for.

## 2. Dashboard

- [x] 2.1 Use team names, never slugs, in every table and card.
- [x] 2.2 Overview and Title odds: correct challenger, comparisons in the reader's units,
      and a decided-title state.
- [x] 2.3 Order selectors and lists by the championship standing.
- [x] 2.4 Replace internal names and model jargon with plain labels.
- [x] 2.5 Season and round pickers that offer only what has been ingested.

## 3. Verification

- [x] 3.1 Unit and API tests for each requirement above; dashboard tests via `AppTest`.
- [x] 3.2 `make lint` and `pytest -m "not network"` pass.
- [x] 3.3 Checked against the live app with `make api` and `make dashboard`.
