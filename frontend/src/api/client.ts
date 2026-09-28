import type {
  AmlDataset,
  ChatResponse,
  ContagionShockResponse,
  FederatedScanResponse,
  GraphSummary,
  MarketGraphResponse,
  MarketShockResponse,
} from "../types/nexusq";

/**
 * Base URL: empty in dev (Vite proxies /api → :8000); in production set
 * VITE_API_BASE to the deployed backend origin, e.g. https://nexus-q-api.onrender.com
 */
const API_BASE = ((import.meta.env.VITE_API_BASE as string | undefined) ?? "").replace(/\/$/, "");

/** Error thrown on non-2xx backend responses. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: string }) => body.detail ?? response.statusText)
      .catch(() => response.statusText);
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

/** Fetch the full interbank topology for the 3D canvas. */
export function fetchGraphSummary(): Promise<GraphSummary> {
  return request<GraphSummary>("/api/graph/summary");
}

/** Trigger a CTQW contagion shock at a bank. */
export function triggerContagionShock(payload: {
  shocked_node_id: number;
  time_t: number;
  gamma?: number | null;
  hops?: number;
  shots?: number | null;
}): Promise<ContagionShockResponse> {
  return request<ContagionShockResponse>("/api/contagion/shock", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Fetch the live-market correlation network (yfinance-backed). */
export function fetchMarketGraph(universe: string, window = 60): Promise<MarketGraphResponse> {
  const params = new URLSearchParams({ universe, window: String(window) });
  return request<MarketGraphResponse>(`/api/market/graph?${params.toString()}`);
}

/** Trigger a CTQW contagion shock on the live market network. */
export function triggerMarketShock(payload: {
  shocked_ticker: string;
  time_t: number;
  gamma?: number | null;
  universe: string;
  window: number;
  shots?: number | null;
}): Promise<MarketShockResponse> {
  return request<MarketShockResponse>("/api/market/contagion/shock", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Trigger the federated QSVC AML scan on the chosen dataset family. */
export function triggerFederatedScan(payload: {
  n_rounds: number;
  learning_rate: number;
  dataset: AmlDataset;
}): Promise<FederatedScanResponse> {
  return request<FederatedScanResponse>("/api/aml/federated-scan", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Ask the analyst copilot about the session's assessments. */
export function sendChat(payload: {
  message: string;
  history?: { role: string; content: string }[];
}): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
