"""L4 — HTTP routes.

The router is intentionally thin: it validates input via pydantic, converts to
the L3 domain types, delegates the actual reasoning to the L5 `Adjudicator`,
and maps the trusted `Decision` back onto the wire response. All wiring of the
concrete adjudicator happens in `app.py` via a dependency override.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from agent import Adjudicator
from server.schemas import (
  AnalyzeRequest,
  AnalyzeResponse,
  HealthResponse,
  RiskEventOut,
)

router = APIRouter()


def get_adjudicator() -> Adjudicator:
  """Dependency placeholder; overridden at app startup with a wired instance."""
  raise RuntimeError("Adjudicator dependency is not configured")


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(adjudicator: Adjudicator = Depends(get_adjudicator)) -> HealthResponse:
  backend = "heuristic" if adjudicator._provider is None else "freellmapi"
  return HealthResponse(status="ok", reasoning_backend=backend)


@router.post("/analyze", response_model=AnalyzeResponse, tags=["pipeline"])
def analyze(
  request: AnalyzeRequest,
  adjudicator: Adjudicator = Depends(get_adjudicator),
) -> AnalyzeResponse:
  decision = adjudicator.decide(request.to_domain(), request.context())
  event = decision.event
  return AnalyzeResponse(
    event=RiskEventOut(
      frame_index=getattr(event, "frame_index", request.frame_index),
      risk=event.risk,
      label=event.label,
      summary=event.summary,
    ),
    sanitized=decision.sanitized,
    used_fallback=decision.used_fallback,
  )
