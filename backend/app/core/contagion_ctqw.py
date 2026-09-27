"""Continuous-Time Quantum Walk (CTQW) contagion engine — Qiskit 1.x, Primitives V2.

Models cascading bank defaults as a probability wave propagating through an
Eisenberg–Noe interbank exposure network.

Mathematical formulation (Farhi–Gutmann style CTQW over the exposure graph):

    H = γ · A_sym            (Hermitian symmetrized adjacency, γ = liquidity velocity)
    U(t) = exp(-i·H·t)       (unitary wave propagation, implemented with PauliEvolutionGate)
    |ψ(0)⟩ = |shocked_bank⟩  (single-excitation computational basis state)

Because A_sym has zero diagonal, the symmetric adjacency decomposes exactly
into two-qubit hopping terms with *no* Jordan–Wigner strings:

    (X_i X_j + Y_i Y_j) / 2  =  |01⟩⟨10| + |10⟩⟨01|   on qubits (i, j)

so the Pauli Hamiltonian reproduces the exact CTQW on the exposure graph and
preserves the one-excitation subspace (the "one defaulted bank at a time"
manifold), up to controllable Trotter error.

Measurement uses ``qiskit.primitives.StatevectorSampler`` (V2). If an IBM
``BackendV2`` is supplied and ``qiskit-ibm-runtime`` is installed, the engine
switches to ``SamplerV2`` with NISQ error-mitigation options (gate twirling +
dynamical decoupling, resilience_level ≥ 1) to demonstrate hardware readiness.

Failure of the quantum path triggers a graceful classical fallback
(Eisenberg–Noe loss cascade blended with PageRank) annotated with
``mode="classical_fallback"``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

from app.core.graph_generator import (
    MAX_SLICE_NODES,
    InterbankSlice,
    degree_centrality_cascade,
    page_rank_cascade,
)

LOGGER = logging.getLogger(__name__)

try:  # Optional hardware path — never required for the demo.
    from qiskit_ibm_runtime import SamplerV2  # type: ignore

    IBM_RUNTIME_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra dep
    SamplerV2 = None  # type: ignore[assignment]
    IBM_RUNTIME_AVAILABLE = False


class CTQWEngineError(RuntimeError):
    """Raised when the CTQW simulation cannot be executed at all."""


@dataclass
class NISQResilienceConfig:
    """Error-mitigation posture applied to V2 primitives (NISQ readiness).

    Attributes:
        resilience_level: V2 primitive resilience level (0 = none, 1 = twirling
            for samplers, ≥2 adds ZNE on estimators).
        dynamical_decoupling: Insert DD pulse sequences on idle qubits.
        gate_twirling: Randomize gates to convert coherent noise into
            stochastic noise (Sampler V2 twirling option).
        optimization_level: Transpiler optimization level for ISA circuits.
    """

    resilience_level: int = 1
    dynamical_decoupling: bool = True
    gate_twirling: bool = True
    optimization_level: int = 3

    def describe(self) -> Dict[str, Any]:
        """JSON-serializable view for the telemetry drawer."""
        return {
            "resilience_level": self.resilience_level,
            "dynamical_decoupling": self.dynamical_decoupling,
            "gate_twirling": self.gate_twirling,
            "optimization_level": self.optimization_level,
            "primitive": "SamplerV2",
            "hardware_target": "simulator (StatevectorSampler)" if not IBM_RUNTIME_AVAILABLE else "backend-aware SamplerV2",
        }


@dataclass
class CTQWContagionResult:
    """Complete result payload for one contagion simulation.

    Attributes:
        mode: "quantum" on success, "classical_fallback" otherwise.
        shocked_node_id: Original graph id of the shocked bank.
        node_probabilities: P(walker at node) from one-hot bitstrings.
        infection_marginals: Blast radius — marginal failure probability per bank.
        histogram: Raw measured bitstring → probability (top-k).
        circuit_depth: Depth of the synthesized (executable) circuit.
        logical_depth: Depth of the abstract evolution circuit.
        n_qubits: Active qubit count (graph slice size).
        n_pauli_terms: Pauli terms in H = γA_sym.
        gamma: Liquidity velocity used.
        time_t: Evolution time used.
        shots: Sampling budget used.
        execution_time_ms: End-to-end execution wall time.
        coherence_leakage: Probability mass lost from the one-excitation
            subspace (Trotter/noise diagnostic).
        qasm3: OpenQASM 3.0 snippet of the executed circuit (truncated).
        telemetry: Primitive/error-mitigation metadata.
        fallback_reason: Why the classical path was taken (if any).
    """

    mode: str
    shocked_node_id: Union[int, str]
    node_probabilities: List[Dict[str, Any]] = field(default_factory=list)
    infection_marginals: Dict[int, float] = field(default_factory=dict)
    histogram: Dict[str, float] = field(default_factory=dict)
    circuit_depth: int = 0
    logical_depth: int = 0
    n_qubits: int = 0
    n_pauli_terms: int = 0
    gamma: float = 0.0
    time_t: float = 0.0
    shots: int = 0
    execution_time_ms: float = 0.0
    coherence_leakage: float = 0.0
    qasm3: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for the FastAPI response body."""
        return {
            "mode": self.mode,
            "shocked_node_id": self.shocked_node_id,
            "node_probabilities": self.node_probabilities,
            "infection_marginals": self.infection_marginals,
            "histogram": self.histogram,
            "circuit_depth": self.circuit_depth,
            "logical_depth": self.logical_depth,
            "n_qubits": self.n_qubits,
            "n_pauli_terms": self.n_pauli_terms,
            "gamma": self.gamma,
            "time_t": self.time_t,
            "shots": self.shots,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "coherence_leakage": round(self.coherence_leakage, 6),
            "qasm3": self.qasm3,
            "telemetry": self.telemetry,
            "fallback_reason": self.fallback_reason,
        }


class CTQWContagionEngine:
    """Simulates financial contagion as a continuous-time quantum walk.

    The interbank exposure matrix is symmetrized into a Hermitian adjacency
    A_sym, mapped to a Pauli Hamiltonian H = γ·A_sym, and evolved with a
    ``PauliEvolutionGate`` starting from the shocked bank's basis state. A V2
    sampler measures the wave function; per-bank failure probabilities are
    derived from the resulting distribution.
    """

    def __init__(
        self,
        gamma: float = 0.8,
        shots: int = 4096,
        trotter_reps: int = 4,
        time_t: float = 1.0,
        resilience: Optional[NISQResilienceConfig] = None,
        seed: int = 2026,
        backend: Any = None,
        qasm_max_chars: int = 2400,
    ) -> None:
        """Configure the engine.

        Args:
            gamma: Liquidity velocity — the CTQW transition rate scaling H.
            shots: V2 sampler measurement budget (4096 keeps p95 < 1.5 s at 10 qubits).
            trotter_reps: Lie–Trotter product-formula repetitions per evolution.
            time_t: Default evolution time (contagion wave propagation horizon).
            resilience: NISQ error-mitigation configuration.
            seed: Deterministic seed for sampler and tie-breaking.
            backend: Optional IBM BackendV2; switches to SamplerV2 + ISA transpile.
            qasm_max_chars: Truncation length for the returned OpenQASM 3 payload.
        """
        if gamma <= 0:
            raise ValueError("gamma (liquidity velocity) must be positive")
        if not 1 <= shots <= 100_000:
            raise ValueError("shots must be within [1, 100000]")
        self.gamma = gamma
        self.shots = shots
        self.trotter_reps = max(1, trotter_reps)
        self.time_t = time_t
        self.resilience = resilience or NISQResilienceConfig()
        self.seed = seed
        self.backend = backend
        self.qasm_max_chars = qasm_max_chars

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------
    @staticmethod
    def symmetrize_exposure_matrix(matrix: Sequence[Sequence[float]]) -> np.ndarray:
        """Return the Hermitian adjacency A_sym = (A + Aᵀ) / 2.

        Bilateral exposures are directional (i lent to j), but the CTQW
        Hamiltonian must be Hermitian; symmetrizing preserves total two-way
        corridor strength while guaranteeing real eigenvalues.

        Args:
            matrix: Directional exposure matrix.

        Returns:
            Symmetric numpy adjacency with zero diagonal.
        """
        a = np.asarray(matrix, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("Exposure matrix must be square")
        np.fill_diagonal(a, 0.0)
        return 0.5 * (a + a.T)

    def build_hamiltonian(self, matrix: Sequence[Sequence[float]], gamma: Optional[float] = None) -> SparsePauliOp:
        """Map the symmetrized exposure matrix to a Pauli Hamiltonian H = γA_sym.

        Each exposure corridor (i, j) with weight w contributes the hopping pair
        γ·w/2·(X_iX_j + Y_iY_j), which is *exactly* the off-diagonal of A_sym in
        the computational basis — no Jordan–Wigner strings required.

        Args:
            matrix: Directional exposure matrix of the graph slice.
            gamma: Optional override of the engine's liquidity velocity.

        Returns:
            A ``SparsePauliOp`` with 2 × n_corridors Pauli terms.
        """
        rate = self.gamma if gamma is None else gamma
        a_sym = self.symmetrize_exposure_matrix(matrix)
        n = a_sym.shape[0]
        if not 2 <= n <= MAX_SLICE_NODES + 1:
            LOGGER.warning("Hamiltonian built for %d qubits (demo budget 4–%d)", n, MAX_SLICE_NODES)

        pauli_terms: List[Tuple[str, List[int], float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                weight = a_sym[i, j]
                if abs(weight) < 1e-12:
                    continue
                coeff = rate * weight / 2.0
                pauli_terms.append(("XX", [i, j], coeff))
                pauli_terms.append(("YY", [i, j], coeff))

        if not pauli_terms:
            raise CTQWEngineError("Graph slice has no exposure corridors; cannot build H")

        return SparsePauliOp.from_sparse_list(pauli_terms, num_qubits=n)

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def build_circuit(
        self,
        matrix: Sequence[Sequence[float]],
        shocked_local_idx: int,
        time_t: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> Tuple[QuantumCircuit, SparsePauliOp]:
        """Build |ψ(0)⟩ = |shocked⟩ evolved under H = γA_sym for time t.

        Args:
            matrix: Directional exposure matrix of the slice.
            shocked_local_idx: Local index of the shocked bank (qubit index).
            time_t: Evolution time; defaults to the engine value.
            gamma: Liquidity velocity; defaults to the engine value.

        Returns:
            Tuple of (measurement-ready circuit, Hamiltonian used).

        Raises:
            CTQWEngineError: If the shock index is out of bounds.
        """
        n = len(matrix)
        if not 0 <= shocked_local_idx < n:
            raise CTQWEngineError(f"shocked index {shocked_local_idx} out of range [0, {n})")

        evolution_time = self.time_t if time_t is None else time_t
        hamiltonian = self.build_hamiltonian(matrix, gamma=gamma)

        circuit = QuantumCircuit(n, name="CTQW-Contagion")
        circuit.x(shocked_local_idx)  # |ψ(0)⟩: the defaulted bank
        circuit.barrier(label="shock")

        evolution = PauliEvolutionGate(
            hamiltonian,
            time=evolution_time,
            synthesis=LieTrotter(reps=self.trotter_reps),
        )
        circuit.append(evolution, range(n))
        circuit.barrier(label="measure")
        circuit.measure_all()

        return circuit, hamiltonian

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _build_sampler(self) -> Tuple[Any, bool]:
        """Instantiate the appropriate V2 sampler.

        Returns:
            Tuple of (sampler instance, needs_isa_transpile flag).
        """
        if self.backend is not None and IBM_RUNTIME_AVAILABLE and SamplerV2 is not None:
            sampler = SamplerV2(mode=self.backend)
            # NISQ error mitigation — explicitly configured V2 primitive options.
            sampler.options.resilience_level = max(0, min(1, self.resilience.resilience_level))
            sampler.options.dynamical_decoupling.enable = self.resilience.dynamical_decoupling
            sampler.options.twirling.enable_gates = self.resilience.gate_twirling
            return sampler, True
        # Simulator default path: exact-statevector sampling, V2 primitive.
        return StatevectorSampler(default_shots=self.shots, seed=self.seed), False

    @staticmethod
    def _extract_counts(pub_result: Any) -> Dict[str, int]:
        """Pull the classical register counts out of a V2 PubResult."""
        data = pub_result.data
        register = getattr(data, "meas", None)
        if register is None:
            raise CTQWEngineError("Sampler PubResult lacks the 'meas' classical register")
        return dict(register.get_counts())

    def _synthesize_qasm(self, circuit: QuantumCircuit) -> Optional[str]:
        """Export the executed circuit as OpenQASM 3.0 (best effort, truncated)."""
        candidates = (circuit, circuit.decompose(reps=1), circuit.decompose(reps=2))
        for candidate in candidates:
            try:
                text = qasm3.dumps(candidate)
                if len(text) > self.qasm_max_chars:
                    text = text[: self.qasm_max_chars] + "\n// … truncated by Nexus-Q telemetry"
                return text
            except Exception:  # noqa: BLE001 — exporter gaps on custom gates
                continue
        LOGGER.warning("OpenQASM 3 export failed for the CTQW circuit")
        return None

    def run(
        self,
        slice_: InterbankSlice,
        shocked_node_id: Union[int, str],
        time_t: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> CTQWContagionResult:
        """Execute one contagion simulation with automatic classical fallback.

        Args:
            slice_: Graph slice containing the shocked bank.
            shocked_node_id: Original (global) id of the shocked bank.
            time_t: Evolution time override.
            gamma: Liquidity velocity override.

        Returns:
            A :class:`CTQWContagionResult` (quantum or classical_fallback mode).
        """
        if shocked_node_id not in slice_.node_ids:
            raise CTQWEngineError(f"Node {shocked_node_id} is not in the active slice")
        shocked_local_idx = slice_.node_ids.index(shocked_node_id)

        started = time.perf_counter()
        try:
            result = self._run_quantum(slice_, shocked_local_idx, time_t, gamma, started)
            result.mode = "quantum"
            return result
        except Exception as exc:  # noqa: BLE001 — resilience contract: never 500 on OOM/sim errors
            LOGGER.exception("CTQW quantum path failed; engaging classical fallback")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return self._classical_fallback(slice_, shocked_local_idx, time_t, elapsed_ms, exc)

    def _run_quantum(
        self,
        slice_: InterbankSlice,
        shocked_local_idx: int,
        time_t: Optional[float],
        gamma: Optional[float],
        started: float,
    ) -> CTQWContagionResult:
        """Quantum execution path (raises on failure; caller falls back)."""
        circuit, hamiltonian = self.build_circuit(
            slice_.matrix, shocked_local_idx, time_t=time_t, gamma=gamma
        )
        evolution_time = self.time_t if time_t is None else time_t

        sampler, needs_transpile = self._build_sampler()
        executable = circuit
        if needs_transpile and self.backend is not None:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            pass_manager = generate_preset_pass_manager(
                optimization_level=self.resilience.optimization_level, backend=self.backend
            )
            executable = pass_manager.run(circuit)

        job = sampler.run([executable])
        raw_counts = self._extract_counts(job.result()[0])

        total = float(sum(raw_counts.values())) or 1.0
        probabilities = {bits: count / total for bits, count in raw_counts.items()}

        # Blast-radius analytics: marginals + one-hot walker distribution.
        n = slice_.n_nodes
        marginals = [0.0] * n
        one_hot_mass = 0.0
        one_hot_probs: Dict[int, float] = {i: 0.0 for i in range(n)}
        for bits, prob in probabilities.items():
            # Qiskit bitstrings are little-endian in text: rightmost char = qubit 0.
            reversed_bits = bits[::-1]
            ones = [idx for idx, ch in enumerate(reversed_bits) if ch == "1"]
            if len(ones) == 1:
                one_hot_mass += prob
                one_hot_probs[ones[0]] += prob
            for idx in ones:
                if idx < n:
                    marginals[idx] += prob

        norm = one_hot_mass or 1.0
        node_probabilities = [
            {
                "node_id": slice_.node_ids[i],
                "label": slice_.labels.get(slice_.node_ids[i], str(slice_.node_ids[i])),
                "walker_probability": round(one_hot_probs[i] / norm, 6),
                "failure_marginal": round(min(1.0, marginals[i]), 6),
                "capital": slice_.capitals[i],
            }
            for i in range(n)
        ]

        synthesized = circuit.decompose(reps=2)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        top_histogram = dict(
            sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:12]
        )

        return CTQWContagionResult(
            mode="quantum",
            shocked_node_id=slice_.node_ids[shocked_local_idx],
            node_probabilities=node_probabilities,
            infection_marginals={
                str(slice_.node_ids[i]): round(min(1.0, marginals[i]), 6) for i in range(n)
            },
            histogram={bits: round(prob, 6) for bits, prob in top_histogram.items()},
            circuit_depth=int(synthesized.depth()),
            logical_depth=int(circuit.depth()),
            n_qubits=n,
            n_pauli_terms=int(len(hamiltonian)),
            gamma=self.gamma if gamma is None else gamma,
            time_t=evolution_time,
            shots=self.shots,
            execution_time_ms=elapsed_ms,
            coherence_leakage=max(0.0, 1.0 - one_hot_mass),
            qasm3=self._synthesize_qasm(circuit),
            telemetry={
                "primitive": "StatevectorSampler (V2)"
                if self.backend is None
                else "qiskit_ibm_runtime.SamplerV2",
                "shots": self.shots,
                "trotter_reps": self.trotter_reps,
                "seed": self.seed,
                "nisq_resilience": self.resilience.describe(),
                "hamiltonian": {
                    "form": "H = γ·A_sym, hopping terms (XX+YY)/2",
                    "pauli_terms": int(len(hamiltonian)),
                    "qubits": n,
                },
            },
        )

    def _classical_fallback(
        self,
        slice_: InterbankSlice,
        shocked_local_idx: int,
        time_t: Optional[float],
        elapsed_ms: float,
        error: Exception,
    ) -> CTQWContagionResult:
        """Graceful degradation: Eisenberg–Noe cascade ∘ PageRank baseline.

        Blends a loss-propagation cascade with PageRank systemic importance so
        the dashboard keeps functioning if the quantum simulation fails.

        Returns:
            A ``CTQWContagionResult`` with ``mode="classical_fallback"``.
        """
        cascade = degree_centrality_cascade(slice_.matrix, slice_.capitals, shocked_local_idx)
        pagerank = page_rank_cascade(slice_.matrix, shocked_local_idx)

        blended = [
            round(min(1.0, 0.6 * cascade[i] + 0.4 * pagerank[i]), 6)
            for i in range(slice_.n_nodes)
        ]
        blended[shocked_local_idx] = 1.0

        node_probabilities = [
            {
                "node_id": slice_.node_ids[i],
                "label": slice_.labels.get(slice_.node_ids[i], str(slice_.node_ids[i])),
                "walker_probability": 0.0,
                "failure_marginal": blended[i],
                "capital": slice_.capitals[i],
            }
            for i in range(slice_.n_nodes)
        ]

        return CTQWContagionResult(
            mode="classical_fallback",
            shocked_node_id=slice_.node_ids[shocked_local_idx],
            node_probabilities=node_probabilities,
            infection_marginals={
                str(slice_.node_ids[i]): blended[i] for i in range(slice_.n_nodes)
            },
            histogram={},
            circuit_depth=0,
            logical_depth=0,
            n_qubits=slice_.n_nodes,
            n_pauli_terms=0,
            gamma=self.gamma,
            time_t=time_t if time_t is not None else self.time_t,
            shots=0,
            execution_time_ms=elapsed_ms,
            coherence_leakage=0.0,
            qasm3=None,
            telemetry={
                "primitive": "classical (NetworkX)",
                "baseline": "Eisenberg–Noe cascade ∘ PageRank blend",
                "nisq_resilience": self.resilience.describe(),
            },
            fallback_reason=f"{type(error).__name__}: {error}",
        )
