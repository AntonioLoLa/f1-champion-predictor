## Purpose

Determine which drivers can still mathematically win the drivers' championship, and which
can do so on their own results alone, using closed-form arithmetic over the remaining
calendar rather than a model.

## Requirements

### Requirement: Elimination compares a ceiling against rivals' current points

A driver SHALL be reported as eliminated only when some rival already holds more points
than that driver could reach by winning every race, sprint and fastest lap still
available. A rival's total can only rise and nothing prevents a rival from scoring nothing
for the rest of the season, so their present total is the only lower bound that exists.

#### Scenario: A large deficit with the season still open is not elimination

- **WHEN** a driver trails by 81 points with 11 rounds still to run
- **THEN** the driver is reported as still in contention

#### Scenario: An unreachable leader eliminates the rest

- **WHEN** the leader's current points exceed a driver's maximum possible total
- **THEN** that driver is reported as eliminated, naming the rival responsible

### Requirement: Self-sufficiency is reported separately from elimination

The system SHALL report, independently of elimination, whether winning every remaining
round would be enough even if the nearest rival finished second in all of them. Failing
this condition MUST NOT be presented as elimination; it means the driver now needs a rival
to drop points.

#### Scenario: A driver who needs help is still alive

- **WHEN** a driver's maximum total beats the leader's current points, but not the leader's
  total after finishing second in every remaining round
- **THEN** the driver is reported as in contention and as not self-sufficient

### Requirement: Ties resolve in the contender's favour

A driver who can draw level with a rival SHALL be treated as still in contention, because
the championship would then be decided on countback, which this capability does not model.

#### Scenario: An exact tie eliminates nobody

- **WHEN** two drivers finish the season on equal points
- **THEN** neither is reported as eliminated

### Requirement: Sprint weekends contribute their extra points

Points still available SHALL account for whether each remaining round is a sprint weekend,
using the points system in force for that season.

#### Scenario: A sprint round raises the ceiling

- **WHEN** the same standings are evaluated with one remaining sprint round instead of one
  conventional round
- **THEN** every driver's maximum possible total is higher

### Requirement: The rounds still to run are reported

The response SHALL state how many rounds of the season remain after the round it was
computed for, so a decided championship can be described without consulting the calendar
separately.

#### Scenario: Remaining rounds are counted after the evaluated round

- **WHEN** contention is computed after round 20 of a 23-round season
- **THEN** the response reports 3 rounds remaining
