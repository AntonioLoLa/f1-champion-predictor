## ADDED Requirements

### Requirement: The rounds still to run are reported

The response SHALL state how many rounds of the season remain after the round it was
computed for, so a decided championship can be described without consulting the calendar
separately.

#### Scenario: Remaining rounds are counted after the evaluated round

- **WHEN** contention is computed after round 20 of a 23-round season
- **THEN** the response reports 3 rounds remaining
