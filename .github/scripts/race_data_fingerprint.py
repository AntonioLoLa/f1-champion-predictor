"""Print a fingerprint of the race data in an f1cp SQLite database.

The data-sync workflow compares this before and after `f1cp data update` to decide
whether there is anything worth committing. The file itself is no use for that: every
ingestion run appends a row to `data_ingestions` (an audit requirement,
openspec/specs/data-ingestion), so `f1cp.db` changes on every run even when no new round
was published. The fingerprint deliberately leaves that table out.

Standard library only, so it runs without the project's virtualenv.

Usage:
    race_data_fingerprint.py DB           row counts of the race tables
    race_data_fingerprint.py DB --latest  "<season> R<round> <name>" of the latest result
"""

from __future__ import annotations

import sqlite3
import sys

_RACE_TABLES = (
    "races",
    "race_results",
    "sprint_results",
    "qualifying_results",
    "driver_standings",
    "constructor_standings",
    "pit_stops",
)


def fingerprint(con: sqlite3.Connection) -> str:
    """Row counts per race table, joined into one comparable string."""
    counts = (
        con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
        for table in _RACE_TABLES
    )
    return "-".join(str(count) for count in counts)


def latest(con: sqlite3.Connection) -> str:
    """The most recent round that has results, for the commit message."""
    row = con.execute(
        "SELECT r.season, r.round, r.name FROM races r "
        "JOIN race_results rr ON rr.race_id = r.id "
        "ORDER BY r.season DESC, r.round DESC LIMIT 1"
    ).fetchone()
    return "none" if row is None else f"{row[0]} R{row[1]} {row[2]}"


def main(argv: list[str]) -> int:
    """Entry point."""
    con = sqlite3.connect(f"file:{argv[1]}?mode=ro", uri=True)
    try:
        print(latest(con) if "--latest" in argv[2:] else fingerprint(con))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
