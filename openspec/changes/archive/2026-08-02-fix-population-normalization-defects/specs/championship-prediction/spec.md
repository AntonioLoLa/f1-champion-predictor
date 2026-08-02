## MODIFIED Requirements

### Requirement: Probabilities are shares of one championship

Exactly one driver wins a season, so served probabilities SHALL be rescaled to sum to one
within a season and round. Both the pre-rescaling and post-rescaling values MUST be
available, so the rescaling is inspectable rather than hidden.

The scored population for a round SHALL be every driver still contesting that
championship, whether or not they started that particular round, so the eventual champion
is never absent from the group being rescaled. A driver who is mathematically eliminated
SHALL have their probability set to zero before the rescale, so eliminated drivers cannot
absorb probability mass that belongs to drivers who can still win.

#### Scenario: A round's probabilities sum to one

- **WHEN** a round is scored
- **THEN** the served probabilities across all drivers in that round sum to one

#### Scenario: A driver who misses a round is still scored

- **WHEN** a driver contesting the championship does not start a given round
- **THEN** that driver still has a row in the scored population for that round, so the
  eventual champion is never dropped from the group being rescaled

#### Scenario: An eliminated driver never receives rescaled probability mass

- **WHEN** a driver is mathematically eliminated from the championship as of a round
- **THEN** that driver's rescaled probability for that round is zero

### Requirement: The scored population is documented

The set of drivers a round is scored over SHALL be knowable from the response, because it
determines the denominator every rescaled probability depends on. Training and serving
SHALL agree on that set, so a live prediction scores exactly the population the model was
evaluated against.

#### Scenario: Serving and training populations agree

- **WHEN** the same round is scored live and looked up in the training dataset
- **THEN** the two populations are identical
