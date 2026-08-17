"""Validate every registered block and the declarative pipeline DAG.

This is the command the Quickstart points at: it loads `config/pipeline.yaml`,
builds the S0->S3 graph through the real `core.graph.build` path (which runs
every structural + type check the executor relies on), and confirms each
referenced block instantiates against its declared port contract.

It imports only the dependency-free core/engine layers, so it stays runnable
in a fresh checkout with no GPU, camera, or LLM backend present.

Usage:
    python tools/validate.py                       # validate config/pipeline.yaml
    python tools/validate.py --spec other.yaml     # validate a different DAG spec
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engines  # noqa: E402,F401  (import registers the S0-S3 vision blocks)
from core.blocks import available  # noqa: E402
from core.graph import build  # noqa: E402


def _to_build_spec(raw: dict) -> dict:
    """Map the pipeline.yaml shape onto what `core.graph.build` expects.

    pipeline.yaml declares edges as `from`/`to`; build() consumes `src`/`dst`.
    Node dicts already match (id/block/config), so they pass through as-is.
    """
    edges = []
    for e in raw.get("edges", []):
        src = e.get("src", e.get("from"))
        dst = e.get("dst", e.get("to"))
        if src is None or dst is None:
            raise ValueError(f"edge missing endpoints: {e!r}")
        edges.append({"src": src, "dst": dst})
    return {"nodes": raw.get("nodes", []), "edges": edges}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VIGIL blocks + pipeline DAG.")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "pipeline.yaml",
        help="Path to a pipeline DAG spec (default: config/pipeline.yaml).",
    )
    args = parser.parse_args()

    import yaml  # lazy: only validation needs a YAML parser

    print(f"Registered blocks: {', '.join(available())}")

    if not args.spec.exists():
        print(f"ERROR: spec not found: {args.spec}")
        return 1

    raw = yaml.safe_load(args.spec.read_text())
    spec = _to_build_spec(raw)

    try:
        graph = build(spec)  # instantiates every node + runs all graph validations
    except Exception as exc:  # noqa: BLE001 - surface any validation failure cleanly
        print(f"INVALID: {type(exc).__name__}: {exc}")
        return 1

    order = graph.topological_order()
    print(f"Spec '{raw.get('name', args.spec.stem)}' v{raw.get('version', '?')} is VALID.")
    print(f"  nodes:          {len(graph.nodes)}  ({', '.join(order)})")
    print(f"  edges:          {len(graph.edges)}")
    print(f"  execution order: {' -> '.join(order)}")
    if raw.get("output"):
        print(f"  graph output:   {raw['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
