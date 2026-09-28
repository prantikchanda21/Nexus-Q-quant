import { useState } from "react";
import type {
  AmlDataset,
  ContagionShockResponse,
  FederatedScanResponse,
  TerminalMode,
} from "../types/nexusq";
import type { NetworkTopology } from "./Network3DCanvas";

interface ShockControlPanelProps {
  mode: TerminalMode;
  summary: NetworkTopology | null;
  /** Currently selected (shocked) node id — bank index or ticker. */
  selectedNode: number | string | null;
  /** Last shock response (drives the blast-radius ranking table). */
  shock: ContagionShockResponse | null;
  /** Last federated scan (drives the AML provenance header + metrics). */
  federated: FederatedScanResponse | null;
  loading: boolean;
  scanning: boolean;
  /** Active market universe + correlation window (market mode only). */
  universe: string;
  windowDays: number;
  onUniverseChange: (universe: string) => void;
  onWindowChange: (days: number) => void;
  onSelectNode: (nodeId: number | string) => void;
  onShockBank: (nodeId: number, timeT: number, gamma: number, hops: number, shots: number) => void;
  onShockMarket: (ticker: string, timeT: number, gamma: number, universe: string, windowDays: number, shots: number) => void;
  onScan: (rounds: number, lr: number, dataset: AmlDataset) => void;
}

const UNIVERSES: [string, string][] = [
  ["nifty50", "NIFTY 50 · stocks + ETFs"],
  ["it", "IT pack + sector ETF"],
  ["banks", "Banks & financials"],
  ["mixedfunds", "Stocks + mutual-fund proxies"],
];

/**
 * Left rail: liquidity-shock injection controls for both terminal modes.
 * Interbank: CTQW over bilateral exposures (hops = slice radius).
 * Market: CTQW over return correlations from yfinance (window = lookback).
 */
export default function ShockControlPanel({
  mode,
  summary,
  selectedNode,
  shock,
  federated,
  loading,
  scanning,
  universe,
  windowDays,
  onUniverseChange,
  onWindowChange,
  onSelectNode,
  onShockBank,
  onShockMarket,
  onScan,
}: ShockControlPanelProps) {
  const [timeT, setTimeT] = useState(1.5);
  const [gamma, setGamma] = useState(mode === "market" ? 1.5 : 0.8);
  const [hops, setHops] = useState(1);
  const [shots, setShots] = useState(4096);
  const [rounds, setRounds] = useState(12);
  const [lr, setLr] = useState(8.0);
  const [dataset, setDataset] = useState<AmlDataset>("elliptic");

  const ranked = shock
    ? [...shock.node_probabilities].sort((a, b) => b.failure_marginal - a.failure_marginal)
    : [];

  const fireShock = () => {
    if (selectedNode === null) return;
    if (mode === "market") {
      onShockMarket(String(selectedNode), timeT, gamma, universe, windowDays, shots);
    } else {
      onShockBank(Number(selectedNode), timeT, gamma, hops, shots);
    }
  };

  return (
    <aside className="flex h-full w-80 flex-col gap-3 overflow-y-auto p-3">
      <section className="panel">
        <header className="panel-header">
          {mode === "market" ? "correlation shock injector" : "liquidity shock injector"}
        </header>
        <div className="flex flex-col gap-3 p-3">
          {mode === "market" && (
            <>
              <label className="text-[11px] text-slate-400">
                universe
                <select
                  className="mt-1 w-full rounded border border-terminal-border bg-black/50 px-2 py-1.5 text-xs text-slate-200"
                  value={universe}
                  onChange={(event) => onUniverseChange(event.target.value)}
                >
                  {UNIVERSES.map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <Slider
                label="correlation window"
                value={windowDays}
                min={20}
                max={250}
                step={10}
                onChange={onWindowChange}
                integer
                suffix="d"
              />
            </>
          )}

          <label className="text-[11px] text-slate-400">
            {mode === "market" ? "shocked asset" : "shocked bank"}
            <select
              className="mt-1 w-full rounded border border-terminal-border bg-black/50 px-2 py-1.5 text-xs text-slate-200"
              value={selectedNode === null ? "" : String(selectedNode)}
              onChange={(event) => {
                if (event.target.value !== "") {
                  const raw = event.target.value;
                  onSelectNode(mode === "market" ? raw : Number(raw));
                }
              }}
              disabled={!summary || summary.nodes.length === 0}
            >
              <option value="" disabled>
                select {mode === "market" ? "ticker…" : "bank…"}
              </option>
              {(summary?.nodes ?? []).map((node) => (
                <option key={String(node.id)} value={String(node.id)}>
                  {node.label}
                  {node.sector ? ` · ${node.sector}` : ` · cap ${node.capital.toFixed(1)}`}
                </option>
              ))}
            </select>
          </label>

          <Slider label="evolution time t" value={timeT} min={0.2} max={4} step={0.1} onChange={setTimeT} suffix="τ" />
          <Slider
            label={mode === "market" ? "correlation velocity γ" : "liquidity velocity γ"}
            value={gamma}
            min={0.1}
            max={3}
            step={0.05}
            onChange={setGamma}
          />
          {mode === "interbank" && (
            <Slider label="slice hops" value={hops} min={1} max={2} step={1} onChange={setHops} integer />
          )}
          <Slider label="shots" value={shots} min={512} max={32768} step={512} onChange={setShots} integer />

          <button className="btn-danger w-full" disabled={loading || selectedNode === null} onClick={fireShock}>
            {loading ? "propagating ψ(t)…" : "inject shock"}
          </button>
        </div>
      </section>

      <section className="panel">
        <header className="panel-header">
          aml federated scan
          {federated && (
            <span
              className={`rounded px-1.5 text-[10px] normal-case tracking-normal ${
                federated.dataset === "elliptic"
                  ? "bg-violet-950 text-violet-300"
                  : "bg-slate-800 text-slate-400"
              }`}
            >
              {federated.dataset === "elliptic" ? "elliptic · real" : "synthetic"}
            </span>
          )}
        </header>
        <div className="flex flex-col gap-3 p-3">
          <Slider label="federation rounds" value={rounds} min={2} max={12} step={1} onChange={setRounds} integer />
          <Slider label="fedsgd η" value={lr} min={0.5} max={10} step={0.5} onChange={setLr} />
          <label className="text-[11px] text-slate-400">
            training data
            <select
              className="mt-1 w-full rounded border border-terminal-border bg-black/50 px-2 py-1.5 text-xs text-slate-200"
              value={dataset}
              onChange={(event) => setDataset(event.target.value as AmlDataset)}
            >
              <option value="elliptic">Elliptic · real Bitcoin (203k txs)</option>
              <option value="synthetic">Synthetic · simulated silos</option>
            </select>
          </label>
          <button
            className="btn-scan w-full"
            disabled={scanning}
            onClick={() => onScan(rounds, lr, dataset)}
          >
            {scanning ? "training qsvc…" : dataset === "elliptic" ? "scan real bitcoin ledger" : "run federated scan"}
          </button>

          {federated && (
            <div className="grid grid-cols-4 gap-1.5 text-center">
              {[
                ["acc", federated.federated_accuracy],
                ["prec", federated.federated_precision],
                ["rec", federated.federated_recall],
                ["f1", federated.federated_f1],
              ].map(([label, value]) => (
                <div key={label as string} className="rounded border border-terminal-border bg-black/40 px-1 py-1.5">
                  <div className="text-[9px] uppercase tracking-wider text-slate-500">{label}</div>
                  <div
                    className={`text-xs font-semibold ${
                      (value as number) >= 0.5 ? "text-emerald-300" : "text-amber-300"
                    }`}
                  >
                    {((value as number) * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {shock && (
        <section className="panel flex-1">
          <header className="panel-header">
            blast radius
            <span className={shock.mode === "quantum" ? "text-emerald-400" : "text-amber-400"}>
              {shock.mode === "quantum" ? "quantum" : "classical fallback"}
            </span>
          </header>
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-slate-500">
                <th className="px-2 py-1 text-left font-normal">{mode === "market" ? "asset" : "bank"}</th>
                <th className="px-2 py-1 text-right font-normal">P(fail)</th>
                <th className="px-2 py-1 text-right font-normal">walker</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((row) => (
                <tr key={String(row.node_id)} className="border-t border-terminal-border/60">
                  <td className="px-2 py-1 text-slate-300">{row.label}</td>
                  <td
                    className={`px-2 py-1 text-right font-semibold ${
                      row.failure_marginal > 0.5 ? "text-rose-400" : "text-slate-300"
                    }`}
                  >
                    {(row.failure_marginal * 100).toFixed(1)}%
                  </td>
                  <td className="px-2 py-1 text-right text-sky-300">
                    {(row.walker_probability * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </aside>
  );
}

/** Compact labeled range slider. */
function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  integer = false,
  suffix = "",
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  integer?: boolean;
  suffix?: string;
}) {
  return (
    <label className="text-[11px] text-slate-400">
      <span className="flex items-center justify-between">
        <span>{label}</span>
        <span className="text-sky-300">
          {integer ? value : value.toFixed(2)}
          {suffix}
        </span>
      </span>
      <input
        type="range"
        className="mt-1 w-full accent-sky-400"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
