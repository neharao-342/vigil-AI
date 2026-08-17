"""L5 — freellmapi reasoning provider.

A thin, dependency-light client that turns validated detections into a
risk-scored, human-readable event by prompting a free-tier LLM gateway
(freellmapi). Configuration is read from the environment:

    FREELLMAPI_BASE_URL   base URL of the gateway
    FREELLMAPI_API_KEY    bearer token (optional for free tier)
    FREELLMAPI_MODEL      model id (default: 'auto')

If the gateway is unreachable or unconfigured, it degrades gracefully to
the heuristic risk score so pipelines never hard-fail on the LLM hop.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass

from agent.provider import heuristic_risk
from engines.types import RiskEvent, ValidatedDetections

logger = logging.getLogger("vigil.agent.freellmapi")


@dataclass
class FreeLlmApiConfig:
    """Runtime config for the freellmapi provider."""

    base_url: str = ""
    api_key: str = ""
    model: str = "auto"
    timeout: float = 15.0

    @classmethod
    def from_env(cls) -> FreeLlmApiConfig:
        return cls(
            base_url=os.getenv("FREELLMAPI_BASE_URL", "").rstrip("/"),
            api_key=os.getenv("FREELLMAPI_API_KEY", ""),
            model=os.getenv("FREELLMAPI_MODEL", "auto"),
            timeout=float(os.getenv("FREELLMAPI_TIMEOUT", "15")),
        )


class FreeLlmApiProvider:
    """ReasoningProvider backed by the freellmapi gateway."""

    def __init__(self, config: FreeLlmApiConfig | None = None) -> None:
        self.config = config or FreeLlmApiConfig.from_env()

    def adjudicate(
        self, detections: ValidatedDetections, context: dict
    ) -> RiskEvent:
        """Return a risk-scored event for the given detections."""
        fallback = self._fallback(detections)
        if not self.config.base_url:
            logger.info("freellmapi unconfigured; using heuristic fallback")
            return fallback
        try:
            payload = self._chat(self._build_prompt(detections, context))
            return self._parse(payload, detections, fallback)
        except Exception as exc:  # noqa: BLE001 (degrade, never crash pipeline)
            logger.warning("freellmapi call failed (%s); using fallback", exc)
            return fallback

    def _build_prompt(
        self, detections: ValidatedDetections, context: dict
    ) -> str:
        items = [
            {"label": d.label, "confidence": d.confidence, "bbox": list(d.bbox)}
            for d in detections.items
        ]
        scene = context.get("scene", "a monitored camera feed")
        return (
            "You are a security vision analyst. Given detections from "
            f"{scene}, return STRICT JSON with keys "
            '"risk" (0..1 float), "label" (short string), "summary" '
            "(one sentence). Respond with ONLY the JSON object — no markdown, "
            "no code fences, no commentary. Detections: "
            + json.dumps(items)
        )

    def _chat(self, prompt: str) -> dict:
        url = f"{self.config.base_url}/v1/chat/completions"
        body = json.dumps(
            {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
        ).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "vigil/0.1"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            return json.loads(resp.read().decode())

    def _parse(
        self,
        payload: dict,
        detections: ValidatedDetections,
        fallback: RiskEvent,
    ) -> RiskEvent:
        try:
            content = payload["choices"][0]["message"]["content"]
            data = _extract_json(content)
            return RiskEvent(
                frame_index=detections.frame_index,
                risk=float(max(0.0, min(1.0, data["risk"]))),
                label=str(data.get("label", fallback.label)),
                summary=str(data.get("summary", fallback.summary)),
                detections=detections.items,
                meta={"provider": "freellmapi", "model": self.config.model},
            )
        except (KeyError, ValueError, TypeError):
            return fallback

    def _fallback(self, detections: ValidatedDetections) -> RiskEvent:
        risk = heuristic_risk(detections)
        label = "activity" if detections.items else "clear"
        summary = (
            f"{len(detections.items)} object(s) detected"
            if detections.items
            else "No objects detected"
        )
        return RiskEvent(
            frame_index=detections.frame_index,
            risk=risk,
            label=label,
            summary=summary,
            detections=detections.items,
            meta={"provider": "heuristic"},
        )


def _extract_json(content: str) -> dict:
    """Extract a JSON object from an LLM reply.

    Real-world chat models often wrap JSON in prose or ```json code fences``` even
    when told not to. Try a direct parse first, then fall back to the outermost
    {...} span. Raises ValueError if nothing parseable is found, so the caller
    degrades to the heuristic instead of crashing.
    """
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        return json.loads(content[start : end + 1])
    raise ValueError("no JSON object found in LLM response")
