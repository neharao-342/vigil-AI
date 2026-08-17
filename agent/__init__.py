"""L5 — Agent (reasoning) layer public API.

Exposes the narrow surface the L4 server depends on:
  * `Adjudicator` / `Decision` — the orchestration entry point,
  * `ReasoningProvider` — the swappable backend protocol,
  * `FreeLlmApiProvider` / `FreeLlmApiConfig` — the default freellmapi backend,
  * `heuristic_risk` — the dependency-free fallback scorer.
"""
from __future__ import annotations

from agent.adjudicator import Adjudicator, Decision
from agent.provider import ReasoningProvider, heuristic_risk
from agent.freellmapi_client import FreeLlmApiConfig, FreeLlmApiProvider
from agent.safety import enforce_output, looks_like_injection, sanitize_text

__all__ = [
  "Adjudicator",
  "Decision",
  "ReasoningProvider",
  "heuristic_risk",
  "FreeLlmApiConfig",
  "FreeLlmApiProvider",
  "enforce_output",
  "looks_like_injection",
  "sanitize_text",
]
