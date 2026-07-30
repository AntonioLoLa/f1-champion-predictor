"""FastAPI routers, one module per resource."""

from f1cp.api.routers import contention, data, explain, health, model, predict, standings, train

__all__ = ["contention", "data", "explain", "health", "model", "predict", "standings", "train"]
