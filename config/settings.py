"""Application settings loaded from the environment.

A single, dependency-light settings object read once at startup. Values map
directly onto the `.env.example` keys so local, docker, and CI environments
share one contract. No secrets are hard-coded; everything falls back to safe
defaults so the app boots even with an empty environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE = REPO_ROOT / "config" / "pipeline.yaml"


def _as_bool(value: str | None, default: bool = False) -> bool:
  if value is None:
    return default
  return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
  """Immutable runtime configuration for the VIGIL service."""

  host: str = "0.0.0.0"
  port: int = 8000
  log_level: str = "info"
  pipeline_path: str = str(DEFAULT_PIPELINE)

  # L5 reasoning backend (empty base_url => heuristic fallback).
  freellmapi_base_url: str = ""
  freellmapi_model: str = "auto"

  # Vision defaults applied when a block omits them.
  min_confidence: float = 0.25
  frame_size: int = 640

  @classmethod
  def from_env(cls) -> "Settings":
    return cls(
      host=os.getenv("VIGIL_HOST", "0.0.0.0"),
      port=int(os.getenv("VIGIL_PORT", "8000")),
      log_level=os.getenv("VIGIL_LOG_LEVEL", "info"),
      pipeline_path=os.getenv("VIGIL_PIPELINE", str(DEFAULT_PIPELINE)),
      freellmapi_base_url=os.getenv("FREELLMAPI_BASE_URL", "").rstrip("/"),
      freellmapi_model=os.getenv("FREELLMAPI_MODEL", "auto"),
      min_confidence=float(os.getenv("VIGIL_MIN_CONFIDENCE", "0.25")),
      frame_size=int(os.getenv("VIGIL_FRAME_SIZE", "640")),
    )

  @property
  def reasoning_backend(self) -> str:
    return "freellmapi" if self.freellmapi_base_url else "heuristic"


def get_settings() -> Settings:
  """Return settings built from the current environment."""
  return Settings.from_env()
