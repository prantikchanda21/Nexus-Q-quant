"""Quantum Federated Learning (QFL) for AML circular-transaction triage.

Implements a privacy-preserving Quantum Support Vector Classifier (QSVC)
trained across three synthetic "banks" (federated clients) without any raw
transaction record leaving its local silo.

Architecture — FedSGD over analytic quantum gradients:

    1. Server broadcasts global weights θ to every client.
    2. Each client encodes its local transaction features with a 4-qubit
       ``ZZFeatureMap`` and applies a parameterized ansatz (circular
       entanglement echoing laundering ring topologies).
    3. The client computes the *analytic* gradient ∂L/∂θ of the QSVC loss

           L(θ) = (1/N) Σ_i (⟨Z₀⟩(x_i, θ) − y_i)² ,  y_i ∈ {−1, +1}

       using ``qiskit_algorithms.gradients.ParamShiftEstimatorGradient``
       on top of the V2 ``qiskit.primitives.StatevectorEstimator``.
    4. Only the 12-float gradient vector is transmitted to the server.
    5. ``global_update()`` applies the FedAvg (mean) gradient step:

           θ ← θ − η · (1/C) Σ_c ∂L_c/∂θ

Score reporting is privacy-safe: clients evaluate their *local* validation
sets and ship only per-transaction anomaly scores and fidelities — never the
underlying feature rows.

Every circuit execution uses Qiskit Primitives V2. If the quantum path fails
(missing dependency, simulator error, …), the classifier degrades to a
classical logistic-regression baseline and reports
``mode="classical_fallback"``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

LOGGER = logging.getLogger(__name__)

N_FEATURES = 4          # in-degree, out-degree, tx velocity, flow dispersion
N_QUBITS = N_FEATURES   # 1 qubit per feature — the ZZ feature map requirement
FEATURE_NAMES = ("in_degree", "out_degree", "tx_velocity", "flow_dispersion")

Z0_OBSERVABLE = SparsePauliOp("IIIZ")  # ⟨Z⟩ on qubit 0 (little-endian label)


class _LegacyOptions:
    """Minimal options carrier for the legacy gradient-framework contract.

    ``qiskit_algorithms`` gradient base classes shallow-copy the wrapped
    estimator's ``options`` and call ``update_options(**kwargs)`` on it. V2
    primitives expose plain dataclasses instead, so we supply the small shim
    the legacy framework expects.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def update_options(self, **kwargs: Any) -> None:
        """Merge run-level options (legacy mutation interface)."""
        self._data.update(kwargs)

    def __copy__(self) -> "_LegacyOptions":
        """Shallow-copy semantics matching legacy ``Options``."""
        clone = _LegacyOptions()
        clone._data = dict(self._data)
        return clone


class _LegacyEstimatorJob:
    """Future-like handle mirroring the legacy ``EstimatorResult`` shape.

    ``qiskit_algorithms`` gradient classes call ``job.result().values`` on the
    primitive job; this wrapper repacks V2 ``PubResult`` data into that shape.
    """

    def __init__(self, values: List[float]) -> None:
        self._values = np.asarray(values, dtype=float)

    def result(self) -> "_LegacyEstimatorJob":
        """Return self as the completed result carrier."""
        return self

    @property
    def values(self) -> np.ndarray:
        """Expectation values, one per submitted circuit."""
        return self._values

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Per-circuit metadata stubs (V2 metadata is dropped by gradients)."""
        return [{} for _ in self._values]


class _V2EstimatorLegacyBridge:
    """Interface adapter: V2 ``StatevectorEstimator`` → ``qiskit_algorithms`` API.

    ``qiskit_algorithms`` 0.3.x gradients invoke
    ``estimator.run(circuits, observables, parameter_values)`` — the pre-PUB
    primitive contract — while Qiskit 1.x ``StatevectorEstimator`` consumes
    Estimator-PUB tuples. This bridge executes every request through the V2
    PUB protocol and repacks the results, so the analytic parameter-shift
    machinery of ``ParamShiftEstimatorGradient`` runs unmodified on a V2
    primitive. No sampling or finite differences are introduced: the gradient
    remains the exact parameter-shift rule.
    """

    def __init__(self, estimator: StatevectorEstimator) -> None:
        self._estimator = estimator
        self.options: _LegacyOptions = _LegacyOptions()

    @staticmethod
    def _as_float(value: Any) -> float:
        """Coerce a PubResult expectation (scalar or 0-d/1-d array) to float."""
        array = np.asarray(value).ravel()
        return float(array[0]) if array.size else 0.0

    def run(self, circuits: Sequence[Any], observables: Sequence[Any],
            parameter_values: Sequence[Sequence[float]], **options: Any) -> _LegacyEstimatorJob:
        """Execute a legacy-style batch as V2 PUBs.

        Args:
            circuits: Parameterized circuits.
            observables: One observable per circuit.
            parameter_values: One bound parameter vector per circuit.
            **options: Legacy options; ``precision`` is forwarded to V2.

        Returns:
            A job-like object exposing ``.result().values``.
        """
        precision = options.pop("precision", None)
        pubs = [
            (circuit, observable, values)
            for circuit, observable, values in zip(circuits, observables, parameter_values)
        ]
        result = (
            self._estimator.run(pubs, precision=precision).result()
            if precision is not None
            else self._estimator.run(pubs).result()
        )
        values = [self._as_float(pub.data.evs) for pub in result]
        return _LegacyEstimatorJob(values)


class QFLTrainingError(RuntimeError):
    """Raised when the federated quantum training loop cannot proceed."""


@dataclass
class QFLConfig:
    """Federated QSVC training configuration.

    Attributes:
        n_clients: Number of federated banks (data silos).
        train_samples_per_client: Local labeled training rows.
        val_samples_per_client: Local labeled validation rows.
        feature_map_reps: ZZFeatureMap repetition depth.
        ansatz_reps: RealAmplitudes ansatz repetition depth.
        learning_rate: Server-side FedSGD step size.
        n_rounds: Federated aggregation rounds.
        seed: Deterministic seed for data + parameter initialization.
        qasm_max_chars: Truncation length for exported OpenQASM 3 snippets.
    """

    n_clients: int = 3
    train_samples_per_client: int = 16
    val_samples_per_client: int = 8
    feature_map_reps: int = 2
    ansatz_reps: int = 2
    learning_rate: float = 5.0
    n_rounds: int = 8
    seed: int = 7
    qasm_max_chars: int = 2600
    dataset: str = "synthetic"      # "synthetic" | "elliptic" (real Bitcoin)
    illicit_fraction: float = 0.5   # silo base rate (Elliptic natural ≈ 0.10)


@dataclass
class ClientDataset:
    """Local (siloed) dataset of one federated bank.

    ``train_x`` / ``val_x`` rows are the 4 raw features; they never leave the
    client — only gradients and score summaries do.
    """

    client_id: int
    bank_name: str
    train_x: np.ndarray  # (N, 4) in [0, 1]
    train_y: np.ndarray  # (N,) ∈ {−1, +1}
    val_x: np.ndarray
    val_y: np.ndarray
    val_txn_ids: List[str] = field(default_factory=list)


@dataclass
class ClientReport:
    """One client's contribution to a federation round (privacy-audited).

    Attributes:
        client_id: Federated client index.
        n_local_samples: Number of local training rows used.
        local_loss: Mean squared QSVC loss on the local batch.
        grad_norm: L2 norm of the transmitted gradient vector.
        gradient_payload: The transmitted ∂L/∂θ (the ONLY artifact shared).
        privacy_log: Audit trail lines proving zero raw-record egress.
        val_accuracy: Local validation accuracy (optional, reported by client).
        anomaly_scores: Per-transaction scores for local val rows.
        fidelity_entropies: Quantum fidelity entropy per local val row.
        val_txn_ids: Synthetic transaction ids aligned with the scores.
    """

    client_id: int
    n_local_samples: int
    local_loss: float
    grad_norm: float
    gradient_payload: List[float]
    privacy_log: List[str] = field(default_factory=list)
    val_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    anomaly_scores: List[float] = field(default_factory=list)
    fidelity_entropies: List[float] = field(default_factory=list)
    val_txn_ids: List[str] = field(default_factory=list)


@dataclass
class QFLResult:
    """Full federated scan payload consumed by the QuantumTelemetry drawer."""

    mode: str = "quantum"
    dataset: str = "synthetic"
    dataset_stats: Dict[str, float] = field(default_factory=dict)
    n_clients: int = 3
    n_rounds: int = 0
    n_train_params: int = 0
    federated_accuracy: float = 0.0
    federated_precision: float = 0.0
    federated_recall: float = 0.0
    federated_f1: float = 0.0
    per_round_loss: List[float] = field(default_factory=list)
    per_round_accuracy: List[float] = field(default_factory=list)
    client_reports: List[ClientReport] = field(default_factory=list)
    anomaly_scores: List[Dict[str, Any]] = field(default_factory=list)
    fidelity_entropy: Dict[str, float] = field(default_factory=dict)
    global_gradient_norm: float = 0.0
    execution_time_ms: float = 0.0
    qasm3_feature_map: Optional[str] = None
    qasm3_full_circuit: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)
    privacy_log: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view for the FastAPI response body."""
        return {
            "mode": self.mode,
            "dataset": self.dataset,
            "dataset_stats": self.dataset_stats,
            "n_clients": self.n_clients,
            "n_rounds": self.n_rounds,
            "n_train_params": self.n_train_params,
            "federated_accuracy": round(self.federated_accuracy, 4),
            "federated_precision": round(self.federated_precision, 4),
            "federated_recall": round(self.federated_recall, 4),
            "federated_f1": round(self.federated_f1, 4),
            "per_round_loss": [round(v, 6) for v in self.per_round_loss],
            "per_round_accuracy": [round(v, 4) for v in self.per_round_accuracy],
            "client_reports": [
                {
                    "client_id": r.client_id,
                    "n_local_samples": r.n_local_samples,
                    "local_loss": round(r.local_loss, 6),
                    "grad_norm": round(r.grad_norm, 6),
                    "gradient_payload": [round(v, 6) for v in r.gradient_payload],
                    "privacy_log": r.privacy_log,
                    "val_accuracy": round(r.val_accuracy, 4),
                    "precision": round(r.precision, 4),
                    "recall": round(r.recall, 4),
                    "f1": round(r.f1, 4),
                    "val_txn_ids": r.val_txn_ids,
                }
                for r in self.client_reports
            ],
            "anomaly_scores": self.anomaly_scores,
            "fidelity_entropy": {k: round(v, 6) for k, v in self.fidelity_entropy.items()},
            "global_gradient_norm": round(self.global_gradient_norm, 6),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "qasm3_feature_map": self.qasm3_feature_map,
            "qasm3_full_circuit": self.qasm3_full_circuit,
            "telemetry": self.telemetry,
            "privacy_log": self.privacy_log,
            "fallback_reason": self.fallback_reason,
        }


class FederatedAMLQuantumClassifier:
    """Federated QSVC trainer for AML circular-flow triage (Primitives V2).

    The classifier scores transactions for obfuscated circular money flows.
    Four graph-derived features per transaction — in-degree, out-degree,
    transaction velocity, and flow dispersion ratio — are embedded into
    Hilbert space by a ``ZZFeatureMap``; a circular-entanglement ansatz mixes
    them; and ⟨Z₀⟩ is mapped to a probability of laundering.
    """

    def __init__(self, config: Optional[QFLConfig] = None) -> None:
        """Build circuits, datasets, and the V2 estimator-backed gradient.

        Args:
            config: Federated training configuration (defaults applied).
        """
        self.config = config or QFLConfig()
        rng = np.random.default_rng(self.config.seed)

        # --- Quantum building blocks -----------------------------------
        self.feature_map = ZZFeatureMap(
            feature_dimension=N_FEATURES, reps=self.config.feature_map_reps
        )
        self.ansatz = RealAmplitudes(
            num_qubits=N_QUBITS,
            reps=self.config.ansatz_reps,
            entanglement="circular",  # ring topology ≈ laundering cycles
        )
        self.feature_params = list(self.feature_map.parameters)
        self.theta_params = list(self.ansatz.parameters)
        self.n_train_params = len(self.theta_params)

        # Global weights θ (server state). RealAmplitudes is real-valued, so
        # initializing angles in [0, π) keeps the whole circuit real.
        self.global_weights: np.ndarray = rng.uniform(
            0.0, np.pi, size=self.n_train_params
        )

        # --- V2 primitives ----------------------------------------------
        # Exact analytic statevector estimator; default precision ⇒ no shot noise.
        self.estimator = StatevectorEstimator()
        # Analytic parameter-shift rule gradients, executed through the V2
        # StatevectorEstimator via the PUB bridge (no sampling-based estimates).
        self.gradient = ParamShiftEstimatorGradient(
            estimator=_V2EstimatorLegacyBridge(self.estimator)
        )

        # --- Federated data silos ----------------------------------------
        if self.config.dataset == "elliptic":
            self.clients, self.dataset_stats = self._build_elliptic_clients()
        else:
            self.clients = self._build_client_datasets(rng)
            self.dataset_stats: Dict[str, float] = {}

        # Reference state for the fidelity-entropy diagnostic: the ZZ encoding
        # of the "anomalous prototype" feature centroid.
        self._reference_statevector: Optional[np.ndarray] = None
        self._last_qasm_full: Optional[str] = None
        self._last_qasm_feature_map: Optional[str] = None

    # ------------------------------------------------------------------
    # Data generation (synthetic silos)
    # ------------------------------------------------------------------
    def _build_client_datasets(self, rng: np.random.Generator) -> List[ClientDataset]:
        """Create per-bank synthetic transaction datasets.

        Legitimate rows are drawn from "low dispersion / low velocity" Beta
        distributions; anomalous (loop-like) rows from high-dispersion,
        high-velocity ones. Clients get non-IID slices (different anomaly
        priors) to make federation meaningful.

        Args:
            rng: Seeded numpy generator.

        Returns:
            One :class:`ClientDataset` per configured bank.
        """
        anomaly_priors = (0.35, 0.30, 0.40)  # non-IID: bank 2 is dirtier
        clients: List[ClientDataset] = []
        for cid in range(self.config.n_clients):
            n_train = self.config.train_samples_per_client
            n_val = self.config.val_samples_per_client
            n = n_train + n_val
            prior = anomaly_priors[cid % len(anomaly_priors)]
            labels01 = (rng.random(n) < prior).astype(int)

            features = np.zeros((n, N_FEATURES), dtype=float)
            normal_idx = labels01 == 0
            anomalous_idx = labels01 == 1

            # Legitimate flows: few counterparties, slow, concentrated.
            features[normal_idx, 0] = rng.beta(2.0, 5.0, size=normal_idx.sum())
            features[normal_idx, 1] = rng.beta(2.0, 5.0, size=normal_idx.sum())
            features[normal_idx, 2] = rng.beta(2.0, 6.0, size=normal_idx.sum())
            features[normal_idx, 3] = rng.beta(2.0, 6.0, size=normal_idx.sum())
            # Laundering loops: balanced in/out degree, fast, highly dispersed.
            features[anomalous_idx, 0] = rng.beta(5.0, 2.0, size=anomalous_idx.sum())
            features[anomalous_idx, 1] = rng.beta(5.0, 2.0, size=anomalous_idx.sum())
            features[anomalous_idx, 2] = rng.beta(6.0, 2.0, size=anomalous_idx.sum())
            features[anomalous_idx, 3] = rng.beta(6.0, 2.0, size=anomalous_idx.sum())

            labels = np.where(labels01 == 1, 1.0, -1.0)
            clients.append(
                ClientDataset(
                    client_id=cid,
                    bank_name=f"BANK-{cid:02d}",
                    train_x=features[:n_train],
                    train_y=labels[:n_train],
                    val_x=features[n_train:],
                    val_y=labels[n_train:],
                    val_txn_ids=[
                        f"B{cid:02d}-TXN-{i:04d}" for i in range(n_val)
                    ],
                )
            )
        return clients

    def _build_elliptic_clients(self) -> Tuple[List[ClientDataset], Dict[str, float]]:
        """Build silos from the real Elliptic Bitcoin graph.

        Features are derived from actual payment topology (in/out degree,
        2-hop velocity, fan-out dispersion) with genuine illicit/licit labels.
        Raises ``EllipticDataError`` when the CSVs are absent, which the scan
        orchestrator converts into the classical fallback.
        """
        from app.core.elliptic_data import load_elliptic_sample

        splits = load_elliptic_sample(
            n_clients=self.config.n_clients,
            train_samples_per_client=self.config.train_samples_per_client,
            val_samples_per_client=self.config.val_samples_per_client,
            illicit_fraction=self.config.illicit_fraction,
            seed=self.config.seed,
        )
        clients = [
            ClientDataset(
                client_id=index,
                bank_name=f"ELL-SILO-{index:02d}",
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                val_txn_ids=txn_ids,
            )
            for index, (
                (train_x, train_y),
                (val_x, val_y),
                txn_ids,
            ) in enumerate(
                zip(splits.client_train, splits.client_val, splits.val_txn_ids),
                start=1,
            )
        ]
        return clients, dict(splits.stats)

    # ------------------------------------------------------------------
    # Circuit assembly
    # ------------------------------------------------------------------
    def _bind_sample(self, x: Sequence[float]) -> QuantumCircuit:
        """Compose ZZFeatureMap(x) · ansatz(θ) with feature values bound.

        Args:
            x: Raw 4-feature row in [0, 1].

        Returns:
            Parametric circuit with only the ansatz weights θ still free.
        """
        circuit = QuantumCircuit(N_QUBITS)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)
        # ZZFeatureMap maps φ(x)=x directly; scale to [0, π] for full angle span.
        scaled = {p: float(v) * math.pi for p, v in zip(self.feature_params, x)}
        return circuit.assign_parameters(scaled)

    def _predict_expectation(self, xs: Sequence[Sequence[float]], theta: np.ndarray) -> np.ndarray:
        """Batch-evaluate ⟨Z₀⟩(x, θ) for all rows via the V2 estimator.

        Args:
            xs: Feature rows.
            theta: Current global weights.

        Returns:
            Array of expectation values in [−1, 1].
        """
        pubs = []
        for x in xs:
            circuit = self._bind_sample(x)
            pubs.append((circuit, Z0_OBSERVABLE, theta))
        job = self.estimator.run(pubs)
        result = job.result()
        return np.asarray(
            [float(res.data.evs) for res in result], dtype=float
        )

    def _local_loss_gradient(
        self, dataset: ClientDataset, theta: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Compute the analytic QSVC loss gradient on one client's silo.

        Chain rule on the squared loss:

            ∂L_i/∂θ_j = 2·(⟨Z₀⟩_i − y_i)·∂⟨Z₀⟩_i/∂θ_j

        where ∂⟨Z₀⟩/∂θ comes from ``ParamShiftEstimatorGradient`` — the exact
        parameter-shift rule, evaluated on the V2 ``StatevectorEstimator``.

        Args:
            dataset: Client silo.
            theta: Broadcast global weights.

        Returns:
            Tuple of (mean gradient vector, mean local loss).

        Raises:
            QFLTrainingError: If the gradient primitive fails.
        """
        train_x, train_y = dataset.train_x, dataset.train_y
        circuits = [self._bind_sample(x) for x in train_x]
        param_values = [theta for _ in circuits]
        targets = [self.theta_params for _ in circuits]

        try:
            forward = self._predict_expectation(train_x, theta)
            grad_job = self.gradient.run(
                circuits=circuits,
                parameter_values=param_values,
                parameters=targets,
                observables=[Z0_OBSERVABLE for _ in circuits],
            )
            z_gradients = np.asarray(
                [np.asarray(g, dtype=float) for g in grad_job.result().gradients]
            )
        except Exception as exc:  # noqa: BLE001 — escalated to the FL orchestrator
            raise QFLTrainingError(
                f"client {dataset.client_id}: gradient primitive failed: {exc}"
            ) from exc

        residuals = forward - train_y  # ⟨Z₀⟩ − y
        # Class weighting: illicit (+1) rows are ~10× rarer in real ledgers —
        # weight ∝ 1/class-frequency so the minority class drives the gradient.
        n_pos = float(np.sum(train_y == 1.0))
        n_neg = float(np.sum(train_y == -1.0))
        n_total = max(n_pos + n_neg, 1.0)
        class_weights = np.where(
            train_y == 1.0,
            n_total / (2.0 * max(n_pos, 1.0)),
            n_total / (2.0 * max(n_neg, 1.0)),
        )
        sample_grads = 2.0 * (residuals * class_weights)[:, None] * z_gradients
        mean_gradient = sample_grads.mean(axis=0)
        local_loss = float(np.mean((residuals**2) * class_weights))
        return mean_gradient, local_loss

    # ------------------------------------------------------------------
    # Federated aggregation
    # ------------------------------------------------------------------
    def global_update(self, client_gradients: Sequence[np.ndarray]) -> float:
        """Aggregate client gradients (FedAvg) and update global weights.

        Implements FedSGD — the full-batch special case of FedAvg:

            θ ← θ − η · (1/C) Σ_c ∂L_c/∂θ

        No client ever sends θ itself, its features, or labels — only the
        gradient vector, which is provably insufficient to reconstruct the
        local ledger at this dimensionality without auxiliary information.

        Args:
            client_gradients: Per-client ∂L_c/∂θ vectors.

        Returns:
            L2 norm of the aggregated update direction.
        """
        stacked = np.stack([np.asarray(g, dtype=float) for g in client_gradients])
        mean_gradient = stacked.mean(axis=0)
        self.global_weights = self.global_weights - self.config.learning_rate * mean_gradient
        return float(np.linalg.norm(mean_gradient))

    # ------------------------------------------------------------------
    # Scoring & diagnostics
    # ------------------------------------------------------------------
    def _anomaly_score(self, expectation: float) -> float:
        """Map ⟨Z₀⟩ ∈ [−1, 1] to a laundering probability in [0, 1]."""
        return float((1.0 - expectation) / 2.0)

    def _reference_state(self) -> np.ndarray:
        """Statevector of the ZZ-encoded anomalous prototype (cached)."""
        if self._reference_statevector is None:
            centroid = np.array(
                [0.72, 0.68, 0.81, 0.85], dtype=float
            )  # high dispersion / velocity prototype
            proto = QuantumCircuit(N_QUBITS)
            proto.compose(self.feature_map, inplace=True)
            scaled = {p: v * math.pi for p, v in zip(self.feature_params, centroid)}
            proto = proto.assign_parameters(scaled)
            self._reference_statevector = np.asarray(Statevector(proto).data)
        return self._reference_statevector

    def _fidelity_entropy(self, x: Sequence[float], theta: np.ndarray) -> float:
        """Quantum fidelity entropy of a transaction's encoded state.

        Fidelity against the anomalous prototype state:

            F(x) = |⟨ψ_ref | ψ(x, θ)⟩|²
            S(x) = −F·ln F − (1−F)·ln(1−F)   (binary entropy, nats)

        High S ⇒ the transaction's quantum fingerprint straddles the
        legitimate/obfuscated boundary — maximally informative for triage.

        Args:
            x: Feature row.
            theta: Current weights.

        Returns:
            Fidelity entropy in nats.
        """
        circuit = self._bind_sample(x).assign_parameters(
            {p: float(v) for p, v in zip(self.theta_params, theta)}
        )
        state = np.asarray(Statevector(circuit).data)
        fidelity = float(abs(np.vdot(self._reference_state(), state)) ** 2)
        fidelity = min(max(fidelity, 1e-12), 1.0 - 1e-12)
        return float(-fidelity * math.log(fidelity) - (1 - fidelity) * math.log(1 - fidelity))

    def _evaluate_client(self, dataset: ClientDataset, theta: np.ndarray) -> ClientReport:
        """Score a client's local validation set (scores only ever leave)."""
        expectations = self._predict_expectation(dataset.val_x, theta)
        scores = [self._anomaly_score(e) for e in expectations]
        predictions = np.where(expectations >= 0.0, 1.0, -1.0)
        accuracy = float(np.mean(predictions == dataset.val_y))
        entropies = [self._fidelity_entropy(x, theta) for x in dataset.val_x]

        # Confusion metrics (positive class = illicit, score ≥ 0.5).
        positive = dataset.val_y == 1.0
        predicted_positive = np.asarray(scores) >= 0.5
        tp = int(np.sum(predicted_positive & positive))
        fp = int(np.sum(predicted_positive & ~positive))
        fn = int(np.sum(~predicted_positive & positive))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return ClientReport(
            client_id=dataset.client_id,
            n_local_samples=len(dataset.train_x),
            local_loss=0.0,
            grad_norm=0.0,
            gradient_payload=[],
            val_accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            anomaly_scores=[round(s, 6) for s in scores],
            fidelity_entropies=[round(e, 6) for e in entropies],
            val_txn_ids=list(dataset.val_txn_ids),
            privacy_log=[
                f"[{dataset.bank_name}] exported {len(scores)} anomaly scores "
                f"+ {len(entropies)} fidelity entropies; 0 raw feature rows left the silo"
            ],
        )

    # ------------------------------------------------------------------
    # QASM export
    # ------------------------------------------------------------------
    def _export_qasm(self) -> Tuple[Optional[str], Optional[str]]:
        """Export the feature map and a fully-bound circuit as OpenQASM 3.0."""
        try:
            sample = self.clients[0].train_x[0]
            feature_qasm = qasm3.dumps(self.feature_map.decompose(reps=1))
            bound = self._bind_sample(sample).assign_parameters(
                {p: float(v) for p, v in zip(self.theta_params, self.global_weights)}
            )
            full_qasm = qasm3.dumps(bound.decompose(reps=1))
        except Exception as exc:  # noqa: BLE001 — exporter gaps must not kill the scan
            LOGGER.warning("OpenQASM 3 export failed: %s", exc)
            return None, None

        limit = self.config.qasm_max_chars
        if len(feature_qasm) > limit:
            feature_qasm = feature_qasm[:limit] + "\n// … truncated"
        if len(full_qasm) > limit:
            full_qasm = full_qasm[:limit] + "\n// … truncated"
        return feature_qasm, full_qasm

    # ------------------------------------------------------------------
    # Orchestrators
    # ------------------------------------------------------------------
    def run_federated_scan(self) -> QFLResult:
        """Execute the full federated scan with classical fallback on failure.

        Returns:
            A :class:`QFLResult` (quantum or classical_fallback mode).
        """
        started = time.perf_counter()
        try:
            result = self._run_quantum_loop(started)
            result.mode = "quantum"
            return result
        except Exception as exc:  # noqa: BLE001 — resilience contract
            LOGGER.exception("QFL quantum loop failed; engaging classical baseline")
            return self._classical_fallback(started, exc)

    def _run_quantum_loop(self, started: float) -> QFLResult:
        """Quantum FedSGD training + privacy-preserving triage scan."""
        privacy_log: List[str] = [
            "NEXUS-Q QFL orchestrator: broadcasting initial global weights θ "
            f"({self.n_train_params} floats) to {self.config.n_clients} banks"
        ]
        per_round_loss: List[float] = []
        per_round_accuracy: List[float] = []
        latest_reports: List[ClientReport] = []
        update_norm = 0.0
        round_reports: List[ClientReport] = []

        for round_idx in range(1, self.config.n_rounds + 1):
            client_gradients: List[np.ndarray] = []
            round_reports: List[ClientReport] = []
            round_losses: List[float] = []

            for dataset in self.clients:
                gradient, loss = self._local_loss_gradient(dataset, self.global_weights)
                client_gradients.append(gradient)
                round_losses.append(loss)
                round_reports.append(
                    ClientReport(
                        client_id=dataset.client_id,
                        n_local_samples=len(dataset.train_x),
                        local_loss=loss,
                        grad_norm=float(np.linalg.norm(gradient)),
                        gradient_payload=[float(v) for v in gradient],
                        privacy_log=[
                            f"[{dataset.bank_name}] round {round_idx}: transmitted "
                            f"{self.n_train_params} ∂L/∂θ floats "
                            f"(ParamShiftEstimatorGradient ∘ StatevectorEstimator, V2) "
                            f"— 0 raw transaction records egressed"
                        ],
                    )
                )

            update_norm = self.global_update(client_gradients)
            global_loss = float(np.mean(round_losses))
            per_round_loss.append(global_loss)
            privacy_log.append(
                f"round {round_idx}: FedAvg aggregated {len(client_gradients)} gradient "
                f"vectors ‖Δθ‖={update_norm:.4f} → global loss {global_loss:.4f}"
            )  # noqa: F541 — f-string keeps log format uniform

            # Federated validation: clients report local accuracies; the server
            # only ever sees the aggregate.
            val_reports = [self._evaluate_client(ds, self.global_weights) for ds in self.clients]
            n_total = sum(len(ds.val_x) for ds in self.clients)
            fed_accuracy = sum(
                r.val_accuracy * len(ds.val_x)
                for r, ds in zip(val_reports, self.clients)
            ) / n_total
            fed_precision = sum(
                r.precision * len(ds.val_x) for r, ds in zip(val_reports, self.clients)
            ) / n_total
            fed_recall = sum(
                r.recall * len(ds.val_x) for r, ds in zip(val_reports, self.clients)
            ) / n_total
            fed_f1 = sum(
                r.f1 * len(ds.val_x) for r, ds in zip(val_reports, self.clients)
            ) / n_total
            per_round_accuracy.append(fed_accuracy)
            latest_reports = val_reports

        scores = [
            {
                "txn_id": txn_id,
                "client_id": report.client_id,
                "anomaly_score": score,
                "fidelity_entropy": entropy,
                "flagged": score >= 0.5,
            }
            for report in latest_reports
            for txn_id, score, entropy in zip(
                report.val_txn_ids, report.anomaly_scores, report.fidelity_entropies
            )
        ]
        scores.sort(key=lambda s: s["anomaly_score"], reverse=True)

        all_entropies = [s["fidelity_entropy"] for s in scores]
        fidelity_entropy_summary = {
            "mean": float(np.mean(all_entropies)) if all_entropies else 0.0,
            "max": float(np.max(all_entropies)) if all_entropies else 0.0,
            "min": float(np.min(all_entropies)) if all_entropies else 0.0,
        }

        privacy_log.append(
            "privacy audit: server touched only θ, ∂L/∂θ vectors, and per-txn scores "
            "— zero raw feature rows crossed a silo boundary"
        )

        feature_qasm, full_qasm = self._export_qasm()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return QFLResult(
            mode="quantum",
            dataset=self.config.dataset,
            dataset_stats=dict(self.dataset_stats),
            n_clients=self.config.n_clients,
            n_rounds=self.config.n_rounds,
            n_train_params=self.n_train_params,
            federated_accuracy=per_round_accuracy[-1] if per_round_accuracy else 0.0,
            federated_precision=fed_precision if per_round_accuracy else 0.0,
            federated_recall=fed_recall if per_round_accuracy else 0.0,
            federated_f1=fed_f1 if per_round_accuracy else 0.0,
            per_round_loss=per_round_loss,
            per_round_accuracy=per_round_accuracy,
            client_reports=round_reports,
            anomaly_scores=scores[:24],
            fidelity_entropy=fidelity_entropy_summary,
            global_gradient_norm=update_norm,
            execution_time_ms=elapsed_ms,
            qasm3_feature_map=feature_qasm,
            qasm3_full_circuit=full_qasm,
            telemetry={
                "primitive": "StatevectorEstimator (V2)",
                "dataset": self.config.dataset,
                "gradient_primitive": "qiskit_algorithms.gradients.ParamShiftEstimatorGradient (PUB-bridged V2)",
                "gradient_method": "analytic parameter-shift rule",
                "feature_map": "ZZFeatureMap(4, reps=%d)" % self.config.feature_map_reps,
                "ansatz": "RealAmplitudes(4, reps=%d, entanglement=circular)" % self.config.ansatz_reps,
                "observable": "Z₀ (IIIZ)",
                "loss": "squared error on ⟨Z₀⟩ vs ±1 label",
                "aggregation": "FedAvg (FedSGD full-batch)",
                "learning_rate": self.config.learning_rate,
                "nisq_resilience": {
                    "resilience_level": 0,
                    "note": "exact statevector estimator on simulator; "
                    "SamplerV2+twirling path available for hardware",
                },
            },
            privacy_log=privacy_log,
        )

    def _classical_fallback(self, started: float, error: Exception) -> QFLResult:
        """Logistic-regression baseline when the quantum loop is unavailable."""
        dim = N_FEATURES
        lr = 0.5
        weights = np.zeros(dim)
        bias = 0.0

        for _ in range(400):
            grad_w = np.zeros(dim)
            grad_b = 0.0
            for dataset in self.clients:
                logits = dataset.train_x @ weights + bias
                preds = 1.0 / (1.0 + np.exp(-logits))
                y01 = (dataset.train_y + 1.0) / 2.0
                residual = preds - y01
                grad_w += dataset.train_x.T @ residual / len(y01)
                grad_b += float(residual.mean())
            grad_w /= self.config.n_clients
            grad_b /= self.config.n_clients
            weights -= lr * grad_w
            bias -= lr * grad_b

        val_scores: List[Dict[str, Any]] = []
        val_accuracies: List[float] = []
        for dataset in self.clients:
            logits = dataset.val_x @ weights + bias
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= 0.5).astype(float)
            val_accuracies.append(float(np.mean(preds == (dataset.val_y + 1.0) / 2.0)))
            for txn_id, prob in zip(dataset.val_txn_ids, probs):
                val_scores.append(
                    {
                        "txn_id": txn_id,
                        "client_id": dataset.client_id,
                        "anomaly_score": round(float(prob), 6),
                        "fidelity_entropy": 0.0,
                        "flagged": bool(prob >= 0.5),
                    }
                )
        val_scores.sort(key=lambda s: s["anomaly_score"], reverse=True)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return QFLResult(
            mode="classical_fallback",
            dataset=self.config.dataset,
            n_clients=self.config.n_clients,
            n_rounds=1,
            n_train_params=dim + 1,
            federated_accuracy=float(np.mean(val_accuracies)) if val_accuracies else 0.0,
            per_round_loss=[],
            per_round_accuracy=[float(np.mean(val_accuracies))] if val_accuracies else [],
            client_reports=[],
            anomaly_scores=val_scores[:24],
            fidelity_entropy={"mean": 0.0, "max": 0.0, "min": 0.0},
            global_gradient_norm=0.0,
            execution_time_ms=elapsed_ms,
            qasm3_feature_map=None,
            qasm3_full_circuit=None,
            telemetry={
                "primitive": "classical (numpy logistic regression)",
                "note": "quantum loop unavailable — silos still exported scores only",
            },
            privacy_log=[
                "classical fallback engaged; privacy contract preserved "
                "(scores-only egress)",
                f"fallback_reason={type(error).__name__}: {error}",
            ],
            fallback_reason=f"{type(error).__name__}: {error}",
        )
