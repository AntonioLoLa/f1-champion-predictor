## Why

A round's served ranking breaks ties between equal probabilities by the order rows happen
to arrive in, which is alphabetical. Once the title is mathematically decided every
eliminated driver is zeroed before the rescale, so the whole field ties at exactly 0 and
the dashboard names whoever sorts first as the "closest challenger": after the 2026
Brazilian GP that was Alexander Albon, with Max Verstappen second in the standings. The
same arbitrary order reaches every list, selector and API caller that sorts by `rank`,
and it is persisted to the `predictions` table.

A prediction also identifies the team only by its slug (`red-bull`), so every consumer
that wants to show it to a person has to resolve the name itself, and none does.

## What Changes

- Ties in the rescaled probability resolve by championship standing at that round
  (better position first, drivers with no position last, then by name), so the ranking
  is deterministic and agrees with the standings wherever the model has no preference.
  Only exact ties are affected: a model that separates two drivers, however slightly,
  keeps its order.
- Each driver in a prediction carries their championship `points` and
  `standing_position` at that round, and the team's display name (`constructor_name`).
- Standings, race results and the season calendar carry the team's display name next to
  its id.
- The standings name each driver's team from their most recent race result. Jolpica's
  standings list every team a driver has raced for that season, in order of first
  appearance, and ingestion stored the last one as the current team: Liam Lawson, back at
  RB after three 2026 rounds standing in at Red Bull, was shown as Red Bull for the rest
  of the season. The feature panel already took the team from the race results, so the
  model was never affected -- only what the standings showed.
- Title contention reports how many rounds remain after the round it was computed for,
  so a consumer can say "decided with three rounds to spare" without a second request.
- All response changes are additive; no field is removed or renamed.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `championship-prediction`: a new requirement on ranking order and on what each driver's
  row identifies.
- `title-contention`: a new requirement that the response states the rounds remaining.
- `results-archive`: a new requirement that standings name the team a driver races for.

## Impact

- `db/repositories/f1_data.py` (`ResultRepository.latest_teams`),
  `data/providers/jolpica.py` (a comment that claimed the opposite),
  `inference/predictor.py` (rank expression, `DriverPrediction`), `api/schemas/*`,
  `api/routers/{predict,standings,archive,contention}.py`, and the dashboard pages that
  consume them.
- Rows persisted to `predictions` get the deterministic rank on their next scoring.
- No retraining: no feature or model input changes.
