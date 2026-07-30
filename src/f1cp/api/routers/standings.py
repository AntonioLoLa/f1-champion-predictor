"""Championship standings endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from f1cp.api.deps import get_database
from f1cp.api.schemas.standings import DriverStandingResponse, SeasonStandingsResponse

if TYPE_CHECKING:
    from f1cp.db.session import Database

__all__ = ["router"]

router = APIRouter(tags=["standings"])


@router.get("/seasons/{season}/standings", response_model=SeasonStandingsResponse)
def standings(
    season: int,
    round: int | None = None,
    database: Database = Depends(get_database),
) -> SeasonStandingsResponse:
    """Return the driver standings after one round of a season.

    Args:
        season: Championship year.
        round: Round within the season. Defaults to the latest round with stored results.
        database: The shared database handle.

    Returns:
        Every driver's points, position and wins after that round.

    Raises:
        HTTPException: 404 if the season or round has no stored standings.
    """
    with database.session_scope() as uow:
        resolved_round = round
        if resolved_round is None:
            rounds = uow.races.rounds_with_results(season)
            if not rounds:
                msg = f"No results stored for season {season}"
                raise HTTPException(status_code=404, detail=msg)
            resolved_round = max(rounds)

        rows = uow.standings.for_round(season, resolved_round)
        if not rows:
            msg = f"No standings for season {season} round {resolved_round}"
            raise HTTPException(status_code=404, detail=msg)

        drivers = {
            driver.driver_id: driver
            for driver in uow.drivers.get_by_ids([row.driver_id for row in rows])
        }
        return SeasonStandingsResponse(
            season=season,
            round=resolved_round,
            standings=[
                DriverStandingResponse(
                    driver_id=row.driver_id,
                    driver_name=(
                        drivers[row.driver_id].full_name
                        if row.driver_id in drivers
                        else row.driver_id
                    ),
                    constructor_id=row.constructor_id,
                    points=row.points,
                    position=row.position,
                    wins=row.wins,
                )
                for row in rows
            ],
        )
