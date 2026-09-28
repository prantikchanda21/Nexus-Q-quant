/**
 * Type contracts mirroring the Nexus-Q FastAPI response schemas.
 * All payloads arrive JSON-serialized from `app.main`.
 */

/** Execution mode: quantum primitive run or graceful classical degradation. */
export type ExecutionMode = "quantum" | "classical_fallback";

/** Terminal data source: synthetic interbank network or live yfinance market. */
export type TerminalMode = "interbank" | "market";

/** One bank's outcome within the contagion blast radius. */
export interface NodeProbability {
  node_id: number | string;
  label: string;
  walker_probability: number;
  failure_marginal: number;
  capital: number;
}

/** Node in the full interbank topology (GET /api/graph/summary). */
export interface GraphNode {
  id: number;
  label: string;
  capital: number;
  degree: number;
}

/** Directed exposure corridor (i lent to j). */
export interface GraphEdge {
  source: number;
  target: number;
  exposure: number;
}

/** Full topology payload for the 3D canvas. */
export interface GraphSummary {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    n_nodes: number;
    n_edges: number;
    density: number;
    avg_clustering: number;
  };
}

/** NISQ error-mitigation posture reported by the V2 primitives. */
export interface NisqResilience {
  resilience_level: number;
  dynamical_decoupling?: boolean;
  gate_twirling?: boolean;
  optimization_level?: number;
  primitive?: string;
  hardware_target?: string;
  note?: string;
}

/** CTQW telemetry block (primitive, shots, Hamiltonian shape). */
export interface ContagionTelemetry {
  primitive: string;
  shots?: number;
  trotter_reps?: number;
  seed?: number;
  nisq_resilience: NisqResilience;
  hamiltonian?: {
    form: string;
    pauli_terms: number;
    qubits: number;
  };
  baseline?: string;
}

/** POST /api/contagion/shock response. */
export interface ContagionShockResponse {
  mode: ExecutionMode;
  shocked_node_id: number | string;
  node_probabilities: NodeProbability[];
  infection_marginals: Record<string, number>;
  histogram: Record<string, number>;
  circuit_depth: number;
  logical_depth: number;
  n_qubits: number;
  n_pauli_terms: number;
  gamma: number;
  time_t: number;
  shots: number;
  execution_time_ms: number;
  coherence_leakage: number;
  qasm3: string | null;
  telemetry: ContagionTelemetry;
  fallback_reason: string | null;
  slice: {
    node_ids: (number | string)[];
    labels: string[];
    capitals: number[];
    edges: { source: number | string; target: number | string; exposure: number }[];
  };
}

/** Node in the live-market correlation network (GET /api/market/graph). */
export interface MarketNode {
  id: string;
  label: string;
  capital: number;
  sector: string;
  degree: number;
}

/** GET /api/market/graph response — correlation-contagion network. */
export interface MarketGraphResponse {
  universe: string;
  universe_label: string;
  window: number;
  degraded: boolean;
  fallback_reason: string | null;
  nodes: MarketNode[];
  edges: GraphEdge[];
  stats: {
    n_assets: number;
    n_edges: number;
    density: number;
    avg_corr: number;
  };
}

/** POST /api/market/contagion/shock response. */
export interface MarketShockResponse extends ContagionShockResponse {
  universe: string;
  degraded_data: boolean;
}

/** QFL client round report (privacy-audited). */
export interface FederatedClientReport {
  client_id: number;
  n_local_samples: number;
  local_loss: number;
  grad_norm: number;
  gradient_payload: number[];
  privacy_log: string[];
  val_accuracy: number;
  val_txn_ids: string[];
}

/** Per-transaction triage record. */
export interface AnomalyScoreRow {
  txn_id: string;
  client_id: number;
  anomaly_score: number;
  fidelity_entropy: number;
  flagged: boolean;
}

/** POST /api/chat response — assistant reply with provenance. */
export interface ChatResponse {
  answer: string;
  mode: "groq" | "local";
  key_label: "primary" | "fallback" | null;
  model: string | null;
  latency_ms: number;
  assessments_in_context: number;
}

/** AML training dataset family. */
export type AmlDataset = "synthetic" | "elliptic";

/** POST /api/aml/federated-scan response. */
export interface FederatedScanResponse {
  mode: ExecutionMode;
  dataset: AmlDataset;
  dataset_stats: Record<string, number>;
  n_clients: number;
  n_rounds: number;
  n_train_params: number;
  federated_accuracy: number;
  federated_precision: number;
  federated_recall: number;
  federated_f1: number;
  per_round_loss: number[];
  per_round_accuracy: number[];
  client_reports: FederatedClientReport[];
  anomaly_scores: AnomalyScoreRow[];
  fidelity_entropy: { mean: number; max: number; min: number };
  global_gradient_norm: number;
  execution_time_ms: number;
  qasm3_feature_map: string | null;
  qasm3_full_circuit: string | null;
  telemetry: {
    primitive: string;
    gradient_primitive: string;
    gradient_method: string;
    feature_map: string;
    ansatz: string;
    observable: string;
    loss: string;
    aggregation: string;
    learning_rate: number;
    nisq_resilience: NisqResilience;
  };
  privacy_log: string[];
  fallback_reason: string | null;
}
