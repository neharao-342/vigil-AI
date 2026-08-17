# Contributing to VIGIL

Thanks for your interest in improving VIGIL. This guide covers how the
project is organized, how to run it locally, and the conventions we follow.

## Architecture at a glance

VIGIL is built as five stacked layers (see `ARCHITECTURE.md`):

- **L1 Core** (`core/`) — block/registry/graph primitives and executor.
- **L2 Graph** (`core/graph/`) — DAG wiring and typed port contracts.
- **L3 Engines** (`engines/`) — vision blocks (Capture, Clean, Detect, Validate) and shared dataclasses.
- **L4 Agent** (`agent/`) — reasoning provider, freellmapi client, safety guardrails, and the Adjudicator.
- **L5 Server + Frontend** (`server/`, `frontend/`) — FastAPI API and the static dashboard.

The pipeline is configured declaratively in `config/pipeline.yaml`.

## Development setup

```bash
git clone https://github.com/shubro18202758/vigil.git
cd vigil
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Useful `make` targets:

```bash
make test     # run pytest
make lint     # run ruff
make run      # start the FastAPI server
```

## Running the stack

1. Copy `.env.example` to `.env` and fill in any keys (a heuristic fallback runs without an LLM backend).
2. Start the API: `uvicorn server.app:create_app --factory --reload`.
3. Open `frontend/index.html` (or serve it) to use the dashboard against `http://localhost:8000`.

## Testing

All changes should keep the suite green:

```bash
pytest -q
```

Add tests under `tests/` mirroring the layer you touch
(`test_engines.py` for L3, `test_agent.py` for L4/L5, etc.). Prefer the
deterministic stub paths so tests stay GPU- and network-free.

## Commit conventions

We use Conventional Commits scoped by layer, e.g.:

- `feat(engines): add optical-flow clean stage`
- `fix(agent): clamp negative risk before enforce_output`
- `test(server): cover /analyze validation errors`
- `ci: cache pip wheels`

## Pull requests

- Keep PRs focused on a single layer/concern where possible.
- Describe how the change maps to the L1-L5 architecture.
- Ensure CI (pytest matrix + ruff) passes before requesting review.

## Safety and secrets

- Never commit real API keys; use `.env` (git-ignored).
- Free-text that reaches the reasoning core must pass through
  `agent.safety.sanitize_text`; model outputs must pass `enforce_output`.
