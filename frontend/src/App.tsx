import { useCallback, useEffect, useState } from "react";
import brandIcon from "./assets/nexusq-icon.png";
import Network3DCanvas, { type NetworkTopology } from "./components/Network3DCanvas";
import ShockControlPanel from "./components/ShockControlPanel";
import QuantumTelemetry from "./components/QuantumTelemetry";
import AssistantChat from "./components/AssistantChat";
import {
  ApiError,
  fetchGraphSummary,
  fetchMarketGraph,
  triggerContagionShock,
  triggerFederatedScan,
  triggerMarketShock,
} from "./api/client";
import type {
  AmlDataset,
  ContagionShockResponse,
  FederatedScanResponse,
  GraphSummary,
  MarketGraphResponse,
  TerminalMode,
} from "./types/nexusq";

/**
 * Nexus-Q financial terminal: 3D network + shock injector + federated AML scan
 * + quantum telemetry. The header toggle switches the whole terminal between
 * the synthetic interbank network and the live yfinance market network — both
 * run the identical CTQW contagion engine on the backend.
 */
export default function App() {
  const [terminalMode, setTerminalMode] = useState<TerminalMode>("interbank");

  // --- Interbank state ----------------------------------------------------
  const [interbank, setInterbank] = useState<GraphSummary | null>(null);
  const [bankShock, setBankShock] = useState<ContagionShockResponse | null>(null);
  const [selectedBank, setSelectedBank] = useState<number | null>(null);

  // --- Market state ---------------------------------------------------------
  const [marketGraph, setMarketGraph] = useState<MarketGraphResponse | null>(null);
  const [marketShock, setMarketShock] = useState<ContagionShockResponse | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [universe, setUniverse] = useState("nifty50");
  const [windowDays, setWindowDays] = useState(60);
  const [marketLoading, setMarketLoading] = useState(false);

  // --- Shared state ---------------------------------------------------------
  const [federated, setFederated] = useState<FederatedScanResponse | null>(null);
  const [loadingShock, setLoadingShock] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [telemetryOpen, setTelemetryOpen] = useState(true);
  const [chatOpen, setChatOpen] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initial interbank topology load.
  useEffect(() => {
    fetchGraphSummary()
      .then((data) => {
        setInterbank(data);
        setSelectedBank(data.nodes[0]?.id ?? null);
      })
      .catch((err: unknown) => setError(describeError(err)));
  }, []);

  // Lazy-load the market network the first time the toggle flips to market.
  useEffect(() => {
    if (terminalMode !== "market" || marketGraph || marketLoading) return;
    setMarketLoading(true);
    setError(null);
    fetchMarketGraph(universe, windowDays)
      .then((data) => {
        setMarketGraph(data);
        setSelectedTicker(data.nodes[0]?.id ?? null);
        if (data.degraded) {
          setError(`live data unavailable — synthetic correlation network in use (${data.fallback_reason})`);
        }
      })
      .catch((err: unknown) => setError(describeError(err)))
      .finally(() => setMarketLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [terminalMode]);

  const refetchMarket = useCallback(
    (nextUniverse: string, nextWindow: number) => {
      setMarketLoading(true);
      setError(null);
      fetchMarketGraph(nextUniverse, nextWindow)
        .then((data) => {
          setMarketGraph(data);
          setMarketShock(null);
          setSelectedTicker(data.nodes[0]?.id ?? null);
          if (data.degraded) {
            setError(`live data unavailable — synthetic correlation network in use (${data.fallback_reason})`);
          }
        })
        .catch((err: unknown) => setError(describeError(err)))
        .finally(() => setMarketLoading(false));
    },
    [],
  );

  const handleUniverseChange = useCallback(
    (next: string) => {
      setUniverse(next);
      refetchMarket(next, windowDays);
    },
    [refetchMarket, windowDays],
  );

  const handleWindowChange = useCallback(
    (days: number) => {
      setWindowDays(days);
      refetchMarket(universe, days);
    },
    [refetchMarket, universe],
  );

  const handleShockBank = useCallback(
    async (nodeId: number, timeT: number, gamma: number, hops: number, shots: number) => {
      setLoadingShock(true);
      setError(null);
      try {
        const response = await triggerContagionShock({
          shocked_node_id: nodeId,
          time_t: timeT,
          gamma,
          hops,
          shots,
        });
        setBankShock(response);
        if (response.mode === "classical_fallback" && response.fallback_reason) {
          setError(`quantum path degraded: ${response.fallback_reason}`);
        }
      } catch (err: unknown) {
        setError(describeError(err));
      } finally {
        setLoadingShock(false);
      }
    },
    [],
  );

  const handleShockMarket = useCallback(
    async (ticker: string, timeT: number, gamma: number, shockUniverse: string, shockWindow: number, shots: number) => {
      setLoadingShock(true);
      setError(null);
      try {
        const response = await triggerMarketShock({
          shocked_ticker: ticker,
          time_t: timeT,
          gamma,
          universe: shockUniverse,
          window: shockWindow,
          shots,
        });
        setMarketShock(response);
        if (response.mode === "classical_fallback" && response.fallback_reason) {
          setError(`quantum path degraded: ${response.fallback_reason}`);
        }
      } catch (err: unknown) {
        setError(describeError(err));
      } finally {
        setLoadingShock(false);
      }
    },
    [],
  );

  const handleScan = useCallback(async (rounds: number, lr: number, dataset: AmlDataset) => {
    setScanning(true);
    setError(null);
    try {
      const response = await triggerFederatedScan({ n_rounds: rounds, learning_rate: lr, dataset });
      setFederated(response);
      if (response.mode === "classical_fallback" && response.fallback_reason) {
        setError(`QFL degraded: ${response.fallback_reason}`);
      }
    } catch (err: unknown) {
      setError(describeError(err));
    } finally {
      setScanning(false);
    }
  }, []);

  const isMarket = terminalMode === "market";
  const activeTopology: NetworkTopology | null = isMarket ? marketGraph : interbank;
  const activeShock: ContagionShockResponse | null = isMarket ? marketShock : bankShock;
  const selectedNode = isMarket ? selectedTicker : selectedBank;
  const busy = loadingShock || marketLoading;

  const handleSelectNode = useCallback(
    (nodeId: number | string) => {
      if (typeof nodeId === "number") setSelectedBank(nodeId);
      else setSelectedTicker(nodeId);
    },
    [],
  );

  const handleNodeClick = useCallback(
    (nodeId: number | string) => {
      handleSelectNode(nodeId);
      // Clicking a node also fires a demo-default shock in the active mode.
      if (isMarket) {
        void handleShockMarket(String(nodeId), 1.5, 1.5, universe, windowDays, 4096);
      } else {
        void handleShockBank(Number(nodeId), 1.5, 0.8, 1, 4096);
      }
    },
    [handleSelectNode, handleShockBank, handleShockMarket, isMarket, universe, windowDays],
  );

  return (
    <div className="app-bg flex h-screen flex-col overflow-hidden">
      <div className="aurora" />
      <div className="hud-grid" />

      <header className="relative flex items-center justify-between border-b border-sky-400/10 bg-[rgba(3,8,18,0.55)] px-4 py-2 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <img
            src={brandIcon}
            alt="Nexus-Q logo"
            className="h-10 w-10 rounded-md object-contain drop-shadow-[0_0_12px_rgba(56,189,248,0.65)]"
          />
          <div className="flex flex-col">
            <h1 className="text-sm font-bold tracking-[0.3em] text-sky-300">
              <span className="[text-shadow:0_0_18px_rgba(56,189,248,0.75)]">NEXUS</span>
              <span className="text-rose-400 [text-shadow:0_0_18px_rgba(244,63,94,0.6)]">-Q</span>
            </h1>
            <span className="text-[9px] font-normal tracking-[0.28em] text-slate-400">
              QUANTUM FINANCIAL TERMINAL
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {activeTopology && (
            <span className="stat-chip">
              {isMarket
                ? `${marketGraph?.stats.n_assets ?? activeTopology.nodes.length} assets · ρ-window ${windowDays}d`
                : `${interbank?.stats.n_nodes ?? activeTopology.nodes.length} banks`}
            </span>
          )}
          {activeShock && (
            <span
              className={`stat-chip ${activeShock.mode === "quantum" ? "text-emerald-300" : "text-amber-300"}`}
            >
              {activeShock.n_qubits}q · {activeShock.execution_time_ms.toFixed(0)} ms
            </span>
          )}

          {/* Interbank ⇄ Market toggle switch */}
          <ModeToggle mode={terminalMode} onChange={setTerminalMode} />
        </div>
      </header>

      {error && (
        <div className="border-b border-amber-500/25 bg-[rgba(43,22,3,0.6)] px-4 py-1.5 text-[11px] text-amber-300 backdrop-blur">
          ⚠ {error}
        </div>
      )}

      <main className="relative flex min-h-0 flex-1">
        <ShockControlPanel
          mode={terminalMode}
          summary={activeTopology}
          selectedNode={selectedNode}
          shock={activeShock}
          federated={federated}
          loading={busy}
          scanning={scanning}
          universe={universe}
          windowDays={windowDays}
          onUniverseChange={handleUniverseChange}
          onWindowChange={handleWindowChange}
          onSelectNode={handleSelectNode}
          onShockBank={handleShockBank}
          onShockMarket={handleShockMarket}
          onScan={handleScan}
        />

        <div className="relative min-w-0 flex-1">
          <Network3DCanvas
            key={terminalMode}
            summary={activeTopology}
            shock={activeShock}
            loading={busy || scanning}
            mode={terminalMode}
            onNodeClick={handleNodeClick}
          />
        </div>

        <QuantumTelemetry
          open={telemetryOpen}
          onToggle={() => setTelemetryOpen((open) => !open)}
          contagion={activeShock}
          federated={federated}
        />

        <AssistantChat open={chatOpen} onToggle={() => setChatOpen((open) => !open)} />
      </main>
    </div>
  );
}

/** Sliding two-state toggle: synthetic interbank network ⇄ live market data. */
function ModeToggle({
  mode,
  onChange,
}: {
  mode: TerminalMode;
  onChange: (mode: TerminalMode) => void;
}) {
  return (
    <div className="flex items-center rounded-full border border-sky-400/20 bg-[rgba(2,8,20,0.6)] p-0.5 text-[11px] shadow-[inset_0_0_12px_rgba(56,189,248,0.08)] backdrop-blur">
      <span className={`px-2 ${mode === "market" ? "text-slate-500" : "text-sky-300"}`}>
        interbank
      </span>
      <button
        role="switch"
        aria-checked={mode === "market"}
        aria-label="Toggle live market mode"
        onClick={() => onChange(mode === "market" ? "interbank" : "market")}
        className={`relative h-5 w-10 rounded-full transition-all duration-300 ${
          mode === "market"
            ? "bg-emerald-500/80 shadow-[0_0_14px_rgba(16,185,129,0.7)]"
            : "bg-slate-700 shadow-[0_0_10px_rgba(56,189,248,0.25)]"
        }`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-300 ${
            mode === "market" ? "translate-x-5" : "translate-x-0.5"
          }`}
        />
      </button>
      <span className={`px-2 ${mode === "market" ? "text-emerald-300" : "text-slate-500"}`}>
        live market
      </span>
    </div>
  );
}

/** Human-readable error extraction across fetch/HTTP failure modes. */
function describeError(err: unknown): string {
  if (err instanceof ApiError) return `API ${err.status}: ${err.message}`;
  if (err instanceof Error) return err.message;
  return String(err);
}
