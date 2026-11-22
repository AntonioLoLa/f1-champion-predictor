## ADDED Requirements

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
