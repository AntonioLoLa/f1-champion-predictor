## ADDED Requirements

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
