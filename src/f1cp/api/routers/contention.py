"""Championship title-contention endpoint: who can still mathematically win."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from f1cp.api.deps import get_database, get_predictor
from f1cp.api.schemas.contention import DriverContentionResponse, TitleContentionResponse

if TYPE_CHECKING:
    from f1cp.db.session import Database
    from f1cp.inference.predictor import PredictionService

__all__ = ["router"]

router = APIRouter(tags=["title-contention"])


@router.get("/seasons/{season}/title-contention", response_model=TitleContentionResponse)
def title_contention(
    season: int,
    round: int | None = None,
    database: Database = Depends(get_database),
    predictor: PredictionService = Depends(get_predictor),
) -> TitleContentionResponse:
    """Determine which drivers can still mathematically win the title.

    A closed-form combinatorial bound, not a machine-learning prediction: every remaining
    round is independent of the others, so the tightest possible check on "can driver X
    still finish ahead of driver Y" has a single optimal scenario (X wins every remaining
    round, which simultaneously caps every rival at second place in those same rounds)
    rather than a space worth searching.

    Args:
        season: Championship year.
        round: Round within the season. Defaults to the latest round with stored results.
        database: The shared database handle.
        predictor: The shared prediction service.

    Returns:
        Every driver with a standing at that round, ranked by current points.

    Raises:
        HTTPException: 404 if the season/round cannot be resolved.
    """
    with database.session_scope() as uow:
        try:
            result = predictor.title_contention(uow, season=season, round_number=round)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        drivers = {
            driver.driver_id: driver
            for driver in uow.drivers.get_by_ids([c.driver_id for c in result.contenders])
        }

    def _name(driver_id: str | None) -> str | None:
        if driver_id is None:
            return None
        driver = drivers.get(driver_id)
        return driver.full_name if driver else driver_id

    return TitleContentionResponse(
        season=result.season,
        round=result.round,
        contenders=[
            DriverContentionResponse(
                driver_id=contender.driver_id,
                driver_name=_name(contender.driver_id) or contender.driver_id,
                current_points=contender.current_points,
                best_case_points=contender.best_case_points,
                still_alive=contender.still_alive,
                controls_own_destiny=contender.controls_own_destiny,
                eliminated_by=contender.eliminated_by,
                eliminated_by_name=_name(contender.eliminated_by),
            )
            for contender in result.contenders
        ],
    )
