"""Nexus-Q FastAPI entrypoint — quantum contagion & AML triage API.

Routes:
    GET  /api/health               — service + subsystem status
    GET  /api/graph/summary        — full interbank topology for the 3D canvas
    POST /api/contagion/shock      — CTQW blast-radius simulation
    POST /api/aml/federated-scan   — federated QSVC triage scan

All quantum execution goes through Primitives V2 with a classical fallback
contract: responses always carry ``mode`` ∈ {"quantum", "classical_fallback"}.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.core.contagion_ctqw import (
    CTQWContagionEngine,
    CTQWEngineError,
    CTQWContagionResult,
    NISQResilienceConfig,
)
from app.core.aml_federated_qsvc import FederatedAMLQuantumClassifier, QFLConfig
from app.core.assessment_log import build_context_digest, record_assessment, recent_events
from app.core.groq_client import CLIENT as GROQ_CLIENT
from app.core.groq_client import GroqError
from app.core.graph_generator import (
    MAX_SLICE_NODES,
    InterbankSlice,
    generate_interbank_graph,
    graph_summary,
    slice_around_node,
)
from app.core.market_network import (
    UNIVERSES,
    MarketDataError,
    MarketNetwork,
    build_market_network,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
LOGGER = logging.getLogger("nexus-q")

N_BANKS = 24
GRAPH_SEED = 42
DEFAULT_GAMMA = 0.8
DEFAULT_SHOTS = 4096

STATE: Dict[str, Any] = {}
MARKET_ENGINE: CTQWContagionEngine = CTQWContagionEngine(
    gamma=1.5, shots=4096, trotter_reps=4
)
MARKET_TIME_T = 1.5


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the interbank universe and the engines once at startup."""
    graph = generate_interbank_graph(n_banks=N_BANKS, seed=GRAPH_SEED)
    STATE["graph"] = graph
    STATE["engine"] = CTQWContagionEngine(
        gamma=DEFAULT_GAMMA,
        shots=4096,
        trotter_reps=4,
        time_t=1.0,
        resilience=NISQResilienceConfig(resilience_level=1, dynamical_decoupling=True, gate_twirling=True),
    )
    STATE["qfl"] = FederatedAMLQuantumClassifier(QFLConfig())
    # Preload the default market network so the first UI toggle is instant;
    # a cold fetch takes ~1-3 s depending on Yahoo latency.
    try:
        STATE["market_default"] = build_market_network(universe="nifty50", window=60, max_assets=10)
        LOGGER.info("market network preloaded (%d assets)", STATE["market_default"].n_assets)
    except Exception as exc:  # noqa: BLE001 — market module is optional at boot
        LOGGER.warning("market network preload deferred: %s", exc)
    LOGGER.info(
        "Nexus-Q ready: %d banks, %d corridors; CTQW engine (γ=%.2f) + QFL QSVC armed",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        DEFAULT_GAMMA,
    )
    yield
    STATE.clear()


app = FastAPI(
    title="Nexus-Q API",
    description="Quantum-Enhanced Financial Contagion & AML Graph Triage Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Local dev origins are always allowed. Production origins (e.g. your Vercel
# domain(s)) come from the ALLOWED_ORIGINS env var as a comma-separated list,
# e.g. ALLOWED_ORIGINS=https://nexus-q.vercel.app,https://nexus-q-git-main.vercel.app
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
_EXTRA_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _EXTRA_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ShockRequest(BaseModel):
    """Body of ``POST /api/contagion/shock``."""

    shocked_node_id: int = Field(..., ge=0, description="Global id of the bank to default (becomes |ψ(0)⟩).")
    time_t: float = Field(1.0, gt=0.0, le=10.0, description="CTQW evolution time (wave propagation horizon).")
    gamma: Optional[float] = Field(None, gt=0.0, le=5.0, description="Liquidity velocity override (γ in H = γA).")
    hops: int = Field(1, ge=1, le=2, description="k-hop slice radius around the shocked bank.")
    shots: Optional[int] = Field(None, ge=256, le=32768, description="V2 sampler shot budget override.")

    @field_validator("gamma", "shots")
    @classmethod
    def _reject_placeholder(cls, value: Optional[float]) -> Optional[float]:
        """Explicitly reject NaN placeholders from the UI layer."""
        if value is None:
            return value
        if isinstance(value, float) and value != value:  # NaN
            raise ValueError("NaN is not a valid override")
        return value


class FederatedScanRequest(BaseModel):
    """Body of ``POST /api/aml/federated-scan``."""

    n_rounds: int = Field(8, ge=1, le=12, description="Federated aggregation rounds.")
    learning_rate: float = Field(5.0, gt=0.0, le=10.0, description="FedSGD step size η.")
    dataset: str = Field(
        "synthetic",
        pattern="^(synthetic|elliptic)$",
        description="synthetic = generated silos; elliptic = real Bitcoin Elliptic dataset.",
    )


class MarketShockRequest(BaseModel):
    """Body of ``POST /api/market/contagion/shock``."""

    shocked_ticker: str = Field(..., min_length=1, description="Ticker to shock, e.g. 'RELIANCE.NS'.")
    time_t: float = Field(1.5, gt=0.0, le=10.0, description="CTQW evolution time.")
    gamma: Optional[float] = Field(None, gt=0.0, le=5.0, description="Correlation velocity override.")
    universe: str = Field("nifty50", description="Universe key (nifty50 | it | banks | mixedfunds).")
    window: int = Field(60, ge=20, le=250, description="Rolling correlation window (trading days).")
    shots: Optional[int] = Field(None, ge=256, le=32768, description="V2 sampler shot budget override.")


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat``."""

    message: str = Field(..., min_length=1, max_length=2000, description="User follow-up question.")
    history: Optional[List[Dict[str, str]]] = Field(
        None, description="Recent conversation turns [{role, content}], oldest first."
    )


class ChatResponse(BaseModel):
    """Assistant reply with provenance metadata."""

    answer: str
    mode: str = Field(..., description="groq | local — local is the offline summarizer.")
    key_label: Optional[str] = Field(None, description="primary | fallback — which Groq key served.")
    model: Optional[str] = None
    latency_ms: float = 0.0
    assessments_in_context: int = 0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Service liveness + subsystem inventory for the terminal header."""
    graph: Optional[Any] = STATE.get("graph")
    return {
        "status": "operational",
        "service": "nexus-q",
        "interbank_banks": graph.number_of_nodes() if graph is not None else 0,
        "engines": {
            "ctqw": "CTQWContagionEngine (StatevectorSampler V2)",
            "aml_qfl": "FederatedAMLQuantumClassifier (StatevectorEstimator V2)",
        },
    }


@app.get("/api/graph/summary")
def get_graph_summary() -> Dict[str, Any]:
    """Full network topology consumed by Network3DCanvas on mount.

    Returns nodes with 3D layout seeds and edges with exposure weights so the
    WebGL force layout can converge instantly on first paint.
    """
    graph = STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    nodes: List[Dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                "label": attrs.get("label", f"BANK-{node_id:03d}"),
                "capital": attrs.get("capital", 10.0),
                "degree": graph.degree(node_id),
            }
        )

    edges: List[Dict[str, Any]] = []
    for src, dst, attrs in graph.edges(data=True):
        edges.append(
            {
                "source": src,
                "target": dst,
                "exposure": attrs.get("exposure", 1.0),
            }
        )

    return {"nodes": nodes, "edges": edges, "stats": graph_summary(graph)}


@app.post("/api/contagion/shock")
def contagion_shock(request: ShockRequest) -> Dict[str, Any]:
    """Run the CTQW contagion simulation for a shocked bank.

    Slices a k-hop neighbourhood of the shocked bank (clamped to the 4–10
    qubit demo budget), evolves |ψ(0)⟩ = |shocked⟩ under H = γA_sym with a
    ``PauliEvolutionGate``, and samples the wave function with the V2
    ``StatevectorSampler``. Falls back to the classical Eisenberg–Noe cascade
    if the quantum path fails.
    """
    graph = STATE.get("graph")
    engine: Optional[CTQWContagionEngine] = STATE.get("engine")
    if graph is None or engine is None:
        raise HTTPException(status_code=503, detail="Engines not initialized")

    try:
        slice_ = slice_around_node(graph, request.shocked_node_id, hops=request.hops)
    except Exception as exc:  # noqa: BLE001 — surfaced as 400 to the dashboard
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Per-request overrides: never leak settings across API calls.
    engine.gamma = request.gamma if request.gamma is not None else DEFAULT_GAMMA
    engine.shots = request.shots if request.shots is not None else DEFAULT_SHOTS

    try:
        result: CTQWContagionResult = engine.run(slice_, request.shocked_node_id, time_t=request.time_t)
    except CTQWEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = result.to_dict()
    payload["slice"] = {
        "node_ids": slice_.node_ids,
        "labels": [slice_.labels[n] for n in slice_.node_ids],
        "capitals": slice_.capitals,
        "edges": [
            {
                "source": slice_.node_ids[i],
                "target": slice_.node_ids[j],
                "exposure": w,
            }
            for i, j, w in slice_.edge_list
        ],
    }
    LOGGER.info(
        "shock @ bank %d → mode=%s, %d qubits, %.1f ms",
        request.shocked_node_id,
        payload["mode"],
        payload["n_qubits"],
        payload["execution_time_ms"],
    )
    payload["assessment_id"] = record_assessment(
        "contagion_shock",
        f"Interbank CTQW shock at node {request.shocked_node_id} ({payload['mode']}, {payload['n_qubits']}q)",
        {
            "shocked_node_id": request.shocked_node_id,
            "mode": payload["mode"],
            "n_qubits": payload["n_qubits"],
            "time_t": payload["time_t"],
            "gamma": payload["gamma"],
            "shots": payload["shots"],
            "logical_depth": payload["logical_depth"],
            "circuit_depth": payload["circuit_depth"],
            "execution_time_ms": payload["execution_time_ms"],
            "coherence_leakage": payload["coherence_leakage"],
            "top_risk": _top_risk_str(payload["node_probabilities"]),
        },
    )
    return payload


def _top_risk_str(rows: List[Dict[str, Any]], top: int = 3) -> str:
    """Compact 'LABEL=pp' list of the highest-risk nodes for the ledger."""
    ranked = sorted(rows, key=lambda r: r["failure_marginal"], reverse=True)[:top]
    return ", ".join(f"{r['label']}={r['failure_marginal']:.2f}" for r in ranked)


@app.get("/api/market/graph")
def market_graph(universe: str = "nifty50", window: int = 60) -> Dict[str, Any]:
    """Live-market correlation network (yfinance) for the 3D canvas.

    Nodes are stocks/ETFs/mutual-fund proxies; edges are return correlations
    (exposure = max(0, ρ)²). Cached with a 15-minute TTL; degrades to a
    deterministic synthetic correlation network if the data provider fails.
    """
    try:
        net = build_market_network(universe=universe, window=window, max_assets=10)
    except MarketDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    degree: Dict[str, int] = {ticker: 0 for ticker in net.node_ids}
    edges = net.edge_rows()
    for edge in edges:
        degree[edge["source"]] += 1  # type: ignore[index]
        degree[edge["target"]] += 1  # type: ignore[index]

    return {
        "universe": net.universe,
        "universe_label": UNIVERSES[universe]["label"],
        "window": net.window,
        "degraded": net.degraded,
        "fallback_reason": net.fallback_reason,
        "nodes": [
            {
                "id": ticker,
                "label": net.labels[ticker],
                "capital": net.capitals[i],
                "sector": net.sectors.get(ticker, "stock"),
                "degree": degree[ticker],
            }
            for i, ticker in enumerate(net.node_ids)
        ],
        "edges": edges,
        "stats": net.stats,
    }


@app.post("/api/market/contagion/shock")
def market_contagion_shock(request: MarketShockRequest) -> Dict[str, Any]:
    """CTQW contagion shock on the live market correlation network.

    Same engine as interbank contagion: |ψ(0)⟩ = |shocked asset⟩ evolves under
    H = γA_sym where A is the correlation-exposure matrix. Returns where the
    distress wave flows — into correlated stocks and the funds tracking them.
    """
    try:
        net = build_market_network(
            universe=request.universe, window=request.window, max_assets=10
        )
    except MarketDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.shocked_ticker not in net.node_ids:
        raise HTTPException(
            status_code=400,
            detail=f"{request.shocked_ticker} is not in the active slice {net.node_ids}",
        )

    slice_ = InterbankSlice(
        node_ids=list(net.node_ids),
        matrix=[row[:] for row in net.matrix],
        capitals=list(net.capitals),
        labels=dict(net.labels),
        edge_list=[
            (i, j, net.matrix[i][j])
            for i in range(net.n_assets)
            for j in range(net.n_assets)
            if i != j and net.matrix[i][j] > 0
        ],
    )

    MARKET_ENGINE.gamma = request.gamma if request.gamma is not None else MARKET_ENGINE.gamma
    MARKET_ENGINE.shots = request.shots if request.shots is not None else MARKET_ENGINE.shots

    result = MARKET_ENGINE.run(slice_, request.shocked_ticker, time_t=request.time_t)
    payload = result.to_dict()
    payload["universe"] = net.universe
    payload["degraded_data"] = net.degraded
    payload["slice"] = {
        "node_ids": slice_.node_ids,
        "labels": [slice_.labels[n] for n in slice_.node_ids],
        "capitals": slice_.capitals,
        "edges": [
            {"source": slice_.node_ids[i], "target": slice_.node_ids[j], "exposure": w}
            for i, j, w in slice_.edge_list
        ],
    }
    LOGGER.info(
        "market shock @ %s [%s] → mode=%s, %d qubits, %.1f ms",
        request.shocked_ticker,
        request.universe,
        payload["mode"],
        payload["n_qubits"],
        payload["execution_time_ms"],
    )
    payload["assessment_id"] = record_assessment(
        "market_shock",
        f"Market CTQW shock at {request.shocked_ticker} ({payload['mode']}, {payload['n_qubits']}q)",
        {
            "shocked_node_id": request.shocked_ticker,
            "universe": request.universe,
            "window": request.window,
            "mode": payload["mode"],
            "n_qubits": payload["n_qubits"],
            "time_t": payload["time_t"],
            "gamma": payload["gamma"],
            "execution_time_ms": payload["execution_time_ms"],
            "degraded_data": payload["degraded_data"],
            "top_risk": _top_risk_str(payload["node_probabilities"]),
        },
    )
    return payload


@app.post("/api/aml/federated-scan")
def federated_scan(request: FederatedScanRequest) -> Dict[str, Any]:
    """Trigger the privacy-preserving federated QSVC triage scan.

    Runs the full FedSGD loop across the three synthetic bank silos using the
    analytic parameter-shift gradient on the V2 ``StatevectorEstimator`` and
    returns the federated accuracy, per-transaction anomaly scores, quantum
    fidelity entropies, and the privacy audit log.
    """
    classifier: Optional[FederatedAMLQuantumClassifier] = STATE.get("qfl")
    if classifier is None:
        raise HTTPException(status_code=503, detail="QFL engine not initialized")

    # Rebuild the engine when the dataset family changes (synthetic ⇄ elliptic).
    if request.dataset != classifier.config.dataset:
        LOGGER.info("rebuilding QFL engine for dataset '%s'", request.dataset)
        classifier = FederatedAMLQuantumClassifier(
            QFLConfig(
                n_rounds=request.n_rounds,
                learning_rate=request.learning_rate,
                dataset=request.dataset,
            )
        )
        STATE["qfl"] = classifier

    classifier.config.n_rounds = request.n_rounds
    classifier.config.learning_rate = request.learning_rate

    result = classifier.run_federated_scan()
    LOGGER.info(
        "federated scan complete → mode=%s, dataset=%s, acc=%.4f, f1=%.4f, %.1f ms",
        result.mode,
        result.dataset,
        result.federated_accuracy,
        result.federated_f1,
        result.execution_time_ms,
    )
    payload = result.to_dict()
    payload["assessment_id"] = record_assessment(
        "federated_scan",
        f"Federated AML scan on {result.dataset} data ({result.mode}, "
        f"acc={result.federated_accuracy:.3f}, F1={result.federated_f1:.3f})",
        {
            "mode": result.mode,
            "dataset": result.dataset,
            "n_clients": result.n_clients,
            "n_rounds": result.n_rounds,
            "learning_rate": classifier.config.learning_rate,
            "n_train_params": result.n_train_params,
            "federated_accuracy": result.federated_accuracy,
            "federated_f1": result.federated_f1,
            "federated_precision": result.federated_precision,
            "federated_recall": result.federated_recall,
            "loss_curve": [round(v, 3) for v in result.per_round_loss],
            "global_gradient_norm": result.global_gradient_norm,
            "execution_time_ms": result.execution_time_ms,
            "n_flagged": sum(1 for s in result.anomaly_scores if s.get("flagged")),
            "fidelity_entropy_mean": result.fidelity_entropy.get("mean", 0.0),
        },
    )
    return payload


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Assistant endpoint: explains the assessments run this session.

    The question is grounded with a digest of the assessment ledger (real
    telemetry from the session's shocks and scans). Tries the primary Groq
    key, then the fallback key, then a deterministic local summarizer — the
    endpoint never 500s on provider failure.
    """
    digest = build_context_digest(limit=6)
    system_prompt = (
        "You are the Nexus-Q analyst copilot inside a quantum finance terminal. "
        "You explain CTQW contagion shocks (interbank and live-market) and "
        "federated QSVC AML scans to a hackathon judge. Ground every claim in "
        "the ASSESSMENT LEDGER below — cite run numbers, parameters, and "
        "numbers verbatim; never invent telemetry that is not there. If asked "
        "about concepts (quantum walks, Trotter error, parameter shift, "
        "FedAvg), explain concisely in 2-4 sentences with one concrete "
        "example from the ledger when possible. Keep answers under 180 words.\n\n"
        f"ASSESSMENT LEDGER (most recent first):\n{digest}"
    )

    history = (request.history or [])[-8:]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        {"role": turn.get("role", "user"), "content": str(turn.get("content", ""))[:1000]}
        for turn in history
        if turn.get("content")
    )
    messages.append({"role": "user", "content": request.message})

    if GROQ_CLIENT.available:
        try:
            result = GROQ_CLIENT.chat(messages)
            return ChatResponse(
                answer=result["answer"],
                mode="groq",
                key_label=result["key_label"],
                model=result["model"],
                latency_ms=round(result["latency_ms"], 1),
                assessments_in_context=len(recent_events(6)),
            )
        except GroqError as exc:
            LOGGER.warning("Groq exhausted (%s); using local summarizer", exc)
            provider_error = str(exc)
    else:
        provider_error = "groq key/package not configured"

    return ChatResponse(
        answer=_local_explanation(request.message, digest),
        mode="local",
        key_label=None,
        model=None,
        latency_ms=0.0,
        assessments_in_context=len(recent_events(6)),
    )


def _local_explanation(question: str, digest: str) -> str:
    """Offline fallback: deterministic briefing from the assessment ledger."""
    lowered = question.lower()
    events = recent_events(3)
    if any(word in lowered for word in ("contagion", "shock", "blast", "wave")):
        focus = [e for e in events if e["kind"] in ("contagion_shock", "market_shock")]
        if focus:
            latest = focus[0]
            return (
                f"[local briefing — Groq unavailable] Latest shock: {latest['summary']}. "
                "The CTQW evolved |ψ(0)⟩ = |shocked asset⟩ under H = γA_sym and sampled "
                "the wave function; failure marginals show where the distress wave "
                f"landed. Details: {latest['details'].get('top_risk', 'n/a')}."
            )
    if any(word in lowered for word in ("aml", "federat", "scan", "launder")):
        focus = [e for e in events if e["kind"] == "federated_scan"]
        if focus:
            latest = focus[0]
            d = latest["details"]
            return (
                f"[local briefing — Groq unavailable] Latest scan: {latest['summary']}. "
                f"Loss curve {d['loss_curve']} over {d['n_rounds']} FedSGD rounds; "
                f"{d['n_flagged']} transactions flagged; only gradient vectors and "
                "scores ever left the silos."
            )
    return (
        "[local briefing — Groq unavailable] I can explain any assessment you've run. "
        "So far this session:\n" + digest[:800]
    )


if __name__ == "__main__":  # pragma: no cover — manual uvicorn convenience
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
