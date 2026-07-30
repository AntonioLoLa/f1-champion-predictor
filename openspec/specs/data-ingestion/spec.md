## Purpose

Acquire Formula 1 results from external providers into the project's own database, so
every downstream stage reads one curated, versioned copy rather than the network.

## Requirements

### Requirement: Ingestion is idempotent

Re-running ingestion over a season already stored SHALL leave the data equivalent, not
duplicated. Writes MUST be keyed on the natural identity of each record.

#### Scenario: Repeating a season does not duplicate rows

- **WHEN** the same season is ingested twice
- **THEN** the stored row counts are unchanged after the second run

### Requirement: Every ingestion run is recorded

Each run SHALL write an audit record of its provider, mode, seasons covered, row counts
and outcome, whether it succeeded or failed. A failure MUST still be recorded and MUST NOT
be swallowed.

#### Scenario: A failing run is still auditable

- **WHEN** a provider raises partway through a run
- **THEN** an audit record with a failed status and the error is stored, and the exception
  propagates

### Requirement: External calls are rate limited and cached

Provider access SHALL respect a configured request rate and MUST reuse a local cache for
seasons treated as immutable, so a re-run does not re-fetch settled history.

#### Scenario: A settled season is served from cache

- **WHEN** a season older than the configured immutability boundary is requested again
- **THEN** no network request is issued for it

### Requirement: Driver and constructor identity is stable across providers

The same competitor appearing under different provider spellings SHALL resolve to one
stored identity, so history joins correctly across sources.

#### Scenario: Two spellings resolve to one driver

- **WHEN** two providers name the same driver differently
- **THEN** results from both attach to a single driver record
