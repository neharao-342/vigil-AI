# VIGIL Architecture

> Visual Intelligence Graph & Inference Layer — a block-graph, real-time vision platform with a swappable free-tier LLM reasoning core (`freellmapi`).

This document describes the system in terms of a **five-layer architecture** and a canonical **S0–S5 pipeline** expressed as a directed acyclic graph (DAG) of typed, validated blocks.

---

## 1. Design Principles

- **Block-graph first.** Every unit of work is a `Block` with typed inputs/outputs. Pipelines are DAGs, validated before execution.
- **Swappable reasoning core.** The LLM layer is an interface, not a vendor. The default implementation targets `freellmapi` (free-tier), but any provider can be dropped in.
- **Real-time by default.** Frames flow through a bounded, back-pressured pipeline; slow stages degrade gracefully instead of blocking capture.
- **Human-readable output.** The system emits risk-scored, explainable events, not raw tensors.
- **Observability everywhere.** Metrics, traces, and structured logs at each layer.

---

## 2. The Five Layers

| Layer | Name | Responsibility | Key Modules |
|-------|------|----------------|-------------|
| L1 | Block Abstraction | Typed block contract, registry, I/O schemas | `core/blocks` |
| L2 | Graph Engine | DAG build, validation, topological order, execution | `core/graph` |
| L3 | Vision Engines | Capture, clean, detect, track, OCR | `engines/` |
| L4 | Serving & UI | API, queue, persistence, dashboard | `server/`, `web/` |
| L5 | Reasoning Core | LLM adjudication, tools, RAG, safety | `agent/` |

### L1 — Block Abstraction
Defines the `Block` interface: a pure, declared transform with a name, input schema, output schema, and config. Blocks self-register into a registry so graphs can be built by reference.

### L2 — Graph Engine
Builds a DAG from a declarative spec, validates it (no cycles, type-compatible edges, all inputs satisfied), computes a topological order, and executes with an executor that supports bounded queues and back-pressure.

### L3 — Vision Engines
Concrete blocks that wrap detection/OCR/tracking models. These are the compute-heavy leaves of the graph and are isolated behind the block contract so they can be swapped or scaled independently.

### L4 — Serving & UI
FastAPI service exposes graph runs, streams events, and persists results. The web dashboard renders the live block-graph (React Flow) and the event feed.

### L5 — Reasoning Core
The agentic layer: consumes structured detections, calls `freellmapi` for adjudication/summarization, applies safety guardrails, and can invoke tools/RAG to produce the final risk-scored, human-readable report.

---

## 3. The S0–S5 Pipeline

The canonical vision-to-report flow is a DAG of six stages:

| Stage | Block | Layer | Input | Output |
|-------|-------|-------|-------|--------|
| S0 | Capture | L3 | RTSP/USB/file source | Frame |
| S1 | Clean | L3 | Frame | NormalizedFrame |
| S2 | Detect | L3 | NormalizedFrame | Detections |
| S3 | Validate | L2/L3 | Detections | ValidatedDetections |
| S4 | Reason | L5 | ValidatedDetections | RiskEvent |
| S5 | Report | L4 | RiskEvent | PersistedEvent + Stream |

### DAG (concept)

```
[S0 Capture] -> [S1 Clean] -> [S2 Detect] -> [S3 Validate] -> [S4 Reason] -> [S5 Report]
                                                 |                              |
                                          (track/OCR fan-out)          (persist + SSE stream)
```

- **S0 Capture** — pulls frames from a source (RTSP, USB, or file) at a bounded rate.
- **S1 Clean** — resize, color-convert, denoise, normalize to model input.
- **S2 Detect** — object detection (YOLO-family) and optional OCR/tracking fan-out.
- **S3 Validate** — schema + confidence + geometry checks; drops or flags invalid detections.
- **S4 Reason** — `freellmapi` adjudicates: classifies risk, summarizes, applies safety guardrails.
- **S5 Report** — persists the event and streams it to the dashboard over SSE/WebSocket.

---

## 4. Reasoning Core Contract (`freellmapi`)

The reasoning core is accessed through a single provider interface so it can be swapped without touching the graph:

```
class ReasoningProvider(Protocol):
    def adjudicate(self, detections: ValidatedDetections, context: dict) -> RiskEvent: ...
```

The default `FreeLlmApiProvider` reads `FREELLMAPI_BASE_URL` and `FREELLMAPI_API_KEY` from the environment (see `.env.example`). Safety guardrails wrap every call: prompt-injection filtering on inputs, and schema + policy validation on outputs.

---

## 5. Runtime Topology

```
            +-------------+        +-------------+
  camera -->|  vigil-api  |<------>|  freellmapi |
            |  (FastAPI)  |        |  (reasoning)|
            +------+------+        +-------------+
                   |
         +---------+---------+
         |                   |
   +-----v-----+       +-----v-----+
   |  postgres |       |  vigil-web|
   | (events)  |       | dashboard |
   +-----------+       +-----------+
         |
   +-----v----------------+
   | prometheus + grafana |
   +----------------------+
```

Services are defined in `docker-compose.yml`. Each layer maps to a module tree under the repo root, and each stage of the S0–S5 pipeline is an independently testable block.

---

## 6. Directory Map

```
core/blocks/   L1  block contract + registry
core/graph/    L2  build, validate, topo, executor
engines/       L3  capture, clean, detect, ocr, track
agent/         L5  freellmapi client, adjudicator, tools, rag, safety
server/        L4  FastAPI routes, queue, db, metrics
web/           L4  dashboard (React Flow event feed)
tools/             bootstrap.py, validate.py
tests/             unit + graph validation tests
```
