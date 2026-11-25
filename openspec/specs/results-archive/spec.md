## Purpose

Serve the historical record as it was ingested — seasons, calendars, race classifications
and both championship tables — so predictions can be read against what actually happened.

## Requirements

### Requirement: The archive is independent of any model

Archive responses SHALL be served from stored results alone and MUST remain available when
no model has been trained.

#### Scenario: The archive answers before any model exists

- **WHEN** the archive is queried against a database with results but no model version
- **THEN** every archive endpoint responds successfully

### Requirement: A calendar includes rounds not yet run

A season's calendar SHALL list every published round, including those without results, so
the shape of the remaining season is visible.

#### Scenario: An unrun round appears without a winner

- **WHEN** a season is partway through
- **THEN** its remaining rounds are listed with no winner recorded

### Requirement: A race reports only a winner it actually has

A round SHALL report a winner only where a driver is classified first. Where no such
classification exists, the system MUST report no winner rather than promoting the
best-placed finisher.

#### Scenario: A race with no classified first place has no winner

- **WHEN** a round's results contain no driver classified first
- **THEN** the round is reported with no winner

### Requirement: A classification preserves its real order

Race results SHALL be ordered as classified, with drivers who did not finish placed after
those who did, rather than sorted by a position they were never awarded.

#### Scenario: Retirements sort last

- **WHEN** a race classification is requested
- **THEN** classified finishers appear before retirements, in finishing order

### Requirement: Standings name the team a driver races for

Each driver's row in the championship standings after a round SHALL name the team of that
driver's most recent Grand Prix up to and including that round, not the team the
standings provider reports, because the provider lists every team a driver has raced for
that season rather than the current one. A driver with no race result that season SHALL
keep the team the provider reports.

#### Scenario: A driver who returns to an earlier team is shown with it

- **WHEN** a driver races for one team, stands in for another for some rounds, and then
  returns to the first
- **THEN** the standings after any later round name the first team

#### Scenario: A driver who misses a round keeps their last team

- **WHEN** a driver does not start a round
- **THEN** the standings after that round name the team of the last race they started
