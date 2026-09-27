"""In-memory assessment ledger: records every analysis the platform runs.

The assistant's grounding source. Each CTQW shock, market shock, and federated
scan appends a structured event; ``build_context_digest`` compiles recent
events into a compact textual brief that is injected into the LLM system
prompt, so the chatbox answers "what did you just do?" questions from real
telemetry rather than hallucination.

Thread-safe via a lock because FastAPI may run endpoints on worker threads.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

_LOCK = threading.Lock()
_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=50)
_SEQ = 0

# Extra headroom (characters) the digest must leave for the answer itself.
_DIGEST_CHAR_BUDGET = 6000


def record_assessment(kind: str, summary: str, details: Dict[str, Any]) -> int:
    """Append one assessment event and return its sequence id.

    Args:
        kind: Event family — "contagion_shock" | "market_shock" | "federated_scan".
        summary: One-line human-readable headline.
        details: Structured payload (mode, qubits, timings, key numbers).
    """
    global _SEQ
    with _LOCK:
        _SEQ += 1
        event = {
            "seq": _SEQ,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "summary": summary,
            "details": details,
        }
        _EVENTS.append(event)
        return _SEQ


def recent_events(limit: int = 8) -> List[Dict[str, Any]]:
    """Most recent assessment events, newest first."""
    with _LOCK:
        return list(_EVENTS)[-limit:][::-1]


def get_event(seq: int) -> Optional[Dict[str, Any]]:
    """Fetch a specific assessment event by id."""
    with _LOCK:
        for event in _EVENTS:
            if event["seq"] == seq:
                return event
        return None


def build_context_digest(limit: int = 6) -> str:
    """Compile recent assessments into a compact brief for the LLM prompt.

    Deterministic, offline, and cheap: this text is what lets the assistant
    narrate the exact runs the user performed — parameters, execution modes,
    qubit counts, blast radii, FedSGD accuracy — without any extra calls.
    """
    events = recent_events(limit)
    if not events:
        return (
            "No assessments have been run yet in this session. "
            "Suggest the user inject a contagion shock or run the federated AML scan."
        )

    lines: List[str] = []
    used = 0
    for event in events:
        d = event["details"]
        if event["kind"] == "contagion_shock":
            line = (
                f"#{event['seq']} [{event['ts']}] INTERBANK CTQW SHOCK — bank {d['shocked_node_id']}: "
                f"mode={d['mode']}, {d['n_qubits']} qubits, t={d['time_t']}, γ={d['gamma']}, "
                f"{d['shots']} shots, depth {d['logical_depth']}→{d['circuit_depth']}, "
                f"{d['execution_time_ms']:.0f} ms, leakage {d['coherence_leakage']:.4f}. "
                f"Top failure marginals: {d['top_risk']}."
            )
        elif event["kind"] == "market_shock":
            line = (
                f"#{event['seq']} [{event['ts']}] MARKET CTQW SHOCK — ticker {d['shocked_node_id']} "
                f"(universe {d['universe']}, ρ-window {d['window']}d): mode={d['mode']}, "
                f"{d['n_qubits']} qubits, t={d['time_t']}, γ={d['gamma']}, "
                f"{d['execution_time_ms']:.0f} ms, degraded_data={d['degraded_data']}. "
                f"Top failure marginals: {d['top_risk']}."
            )
        else:  # federated_scan
            line = (
                f"#{event['seq']} [{event['ts']}] FEDERATED AML SCAN — mode={d['mode']}, "
                f"{d['n_clients']} banks × {d['n_rounds']} FedSGD rounds (η={d['learning_rate']}), "
                f"{d['n_train_params']} ansatz params, final accuracy {d['federated_accuracy']:.3f}, "
                f"loss curve {d['loss_curve']}, ‖Δθ‖={d['global_gradient_norm']:.4f}, "
                f"{d['execution_time_ms']:.0f} ms. Flagged {d['n_flagged']} transactions; "
                f"fidelity entropy mean {d['fidelity_entropy_mean']:.3f} nats."
            )

        if used + len(line) > _DIGEST_CHAR_BUDGET:
            lines.append("…(older assessments omitted)")
            break
        lines.append(line)
        used += len(line)

    return "\n".join(lines)
