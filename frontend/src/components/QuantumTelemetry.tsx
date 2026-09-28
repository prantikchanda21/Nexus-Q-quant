import { useMemo, useState } from "react";
import brandIcon from "../assets/nexusq-icon.png";
import type { ContagionShockResponse, FederatedScanResponse } from "../types/nexusq";

interface QuantumTelemetryProps {
  open: boolean;
  onToggle: () => void;
  contagion: ContagionShockResponse | null;
  federated: FederatedScanResponse | null;
}

type Tab = "histogram" | "qasm" | "nisq";

/**
 * Bottom drawer: raw quantum execution telemetry — V2 primitive histograms,
 * OpenQASM 3.0 of the executed circuits, and the active NISQ resilience
 * posture. Everything shown is straight from the backend payloads.
 */
export default function QuantumTelemetry({
  open,
  onToggle,
  contagion,
  federated,
}: QuantumTelemetryProps) {
  const [tab, setTab] = useState<Tab>("histogram");

  const histogramRows = useMemo(() => {
    if (!contagion) return [];
    return Object.entries(contagion.histogram)
      .map(([bits, prob]) => ({ bits, prob }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 10);
  }, [contagion]);

  const qasm = contagion?.qasm3 ?? federated?.qasm3_full_circuit ?? null;
  const nisq = contagion?.telemetry.nisq_resilience ?? federated?.telemetry.nisq_resilience ?? null;
  const primitive =
    contagion?.telemetry.primitive ?? federated?.telemetry.primitive ?? "—";

  const maxProb = histogramRows[0]?.prob ?? 1;

  return (
    <div
      className={`pointer-events-auto absolute inset-x-0 bottom-0 z-20 transform transition-transform duration-300 ${
        open ? "translate-y-0" : "translate-y-[calc(100%-2.25rem)]"
      }`}
    >
      <div className="panel mb-3 ml-3 flex h-72 flex-col overflow-hidden [margin-right:25rem] max-xl:[margin-right:1rem]">
        <header className="panel-header cursor-pointer select-none" onClick={onToggle}>
          <span className="flex items-center gap-3">
            <span className="flex items-center gap-2 text-sky-300">
              <img
                src={brandIcon}
                alt=""
                className="h-4 w-4 rounded-sm object-contain drop-shadow-[0_0_6px_rgba(56,189,248,0.55)]"
              />
              quantum telemetry
            </span>
            <span className="text-[10px] normal-case tracking-normal text-slate-500">
              {primitive}
            </span>
            {contagion && (
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] ${
                  contagion.mode === "quantum"
                    ? "bg-emerald-950 text-emerald-300"
                    : "bg-amber-950 text-amber-300"
                }`}
              >
                {contagion.mode}
              </span>
            )}
            {federated && (
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] ${
                  federated.mode === "quantum"
                    ? "bg-emerald-950 text-emerald-300"
                    : "bg-amber-950 text-amber-300"
                }`}
              >
                qfl: {federated.mode}
              </span>
            )}
          </span>
          <span className="text-slate-500">{open ? "▾ collapse" : "▴ expand"}</span>
        </header>

        <nav className="flex gap-1 border-b border-terminal-border px-3 py-1.5 text-[11px]">
          {(
            [
              ["histogram", "sampler histogram"],
              ["qasm", "openqasm 3.0"],
              ["nisq", "nisq resilience"],
            ] as [Tab, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`rounded px-2 py-1 uppercase tracking-wider ${
                tab === key ? "bg-sky-950/70 text-sky-300" : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="min-h-0 flex-1 overflow-y-auto p-3 text-[11px] leading-relaxed">
          {tab === "histogram" && (
            <div className="flex flex-col gap-4">
              <section>
                <h4 className="mb-1 text-slate-400">
                  StatevectorSampler (V2) — P(bitstring) after CTQW evolution
                </h4>
                {histogramRows.length === 0 && (
                  <p className="text-slate-600">run a contagion shock to populate the histogram</p>
                )}
                {histogramRows.map(({ bits, prob }) => (
                  <div key={bits} className="flex items-center gap-2">
                    <code className="w-24 text-sky-300">|{bits}⟩</code>
                    <div className="h-2 flex-1 overflow-hidden rounded bg-black/50">
                      <div
                        className="h-full bg-gradient-to-r from-sky-600 to-sky-300"
                        style={{ width: `${(prob / maxProb) * 100}%` }}
                      />
                    </div>
                    <span className="w-16 text-right text-slate-300">{(prob * 100).toFixed(2)}%</span>
                  </div>
                ))}
              </section>

              {federated && (
                <section>
                  <h4 className="mb-1 text-slate-400">
                    FedSGD convergence — loss / federated accuracy per round
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {federated.per_round_loss.map((loss, i) => (
                      <span key={i} className="stat-chip">
                        r{i + 1}: L={loss.toFixed(3)} · acc=
                        {(federated.per_round_accuracy[i] * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                  <p className="mt-2 text-slate-500">
                    global ‖Δθ‖ = {federated.global_gradient_norm.toFixed(4)} ·
                    fidelity-entropy mean {federated.fidelity_entropy.mean.toFixed(3)} nats ·
                    exec {federated.execution_time_ms.toFixed(0)} ms
                  </p>
                </section>
              )}
            </div>
          )}

          {tab === "qasm" && (
            <div>
              <h4 className="mb-1 text-slate-400">OpenQASM 3.0 — executed circuit (truncated)</h4>
              {qasm ? (
                <pre className="max-h-44 overflow-auto rounded border border-terminal-border bg-black/60 p-2 text-[10px] text-emerald-300/90">
                  {qasm}
                </pre>
              ) : (
                <p className="text-slate-600">no circuit exported (classical fallback has no QASM)</p>
              )}
            </div>
          )}

          {tab === "nisq" && (
            <div className="flex flex-col gap-3">
              <section>
                <h4 className="mb-1 text-slate-400">active error-mitigation configuration</h4>
                {nisq ? (
                  <ul className="flex flex-col gap-1 text-slate-300">
                    <li>resilience_level: <span className="text-sky-300">{nisq.resilience_level}</span></li>
                    {nisq.dynamical_decoupling !== undefined && (
                      <li>dynamical decoupling: <span className="text-sky-300">{String(nisq.dynamical_decoupling)}</span></li>
                    )}
                    {nisq.gate_twirling !== undefined && (
                      <li>gate twirling: <span className="text-sky-300">{String(nisq.gate_twirling)}</span></li>
                    )}
                    {nisq.optimization_level !== undefined && (
                      <li>transpiler optimization_level: <span className="text-sky-300">{nisq.optimization_level}</span></li>
                    )}
                    {nisq.hardware_target && (
                      <li>hardware target: <span className="text-sky-300">{nisq.hardware_target}</span></li>
                    )}
                    {nisq.note && <li className="text-slate-500">{nisq.note}</li>}
                  </ul>
                ) : (
                  <p className="text-slate-600">no execution telemetry yet</p>
                )}
              </section>

              {contagion?.telemetry.hamiltonian && (
                <section>
                  <h4 className="mb-1 text-slate-400">hamiltonian</h4>
                  <p className="text-slate-300">
                    {contagion.telemetry.hamiltonian.form} · {contagion.telemetry.hamiltonian.pauli_terms}{" "}
                    Pauli terms · {contagion.telemetry.hamiltonian.qubits} qubits · γ=
                    {contagion.gamma} · t={contagion.time_t} · {contagion.shots} shots ·
                    depth {contagion.logical_depth}→{contagion.circuit_depth} · leakage{" "}
                    {(contagion.coherence_leakage * 100).toFixed(2)}%
                  </p>
                </section>
              )}

              {federated && (
                <section>
                  <h4 className="mb-1 text-slate-400">privacy audit log</h4>
                  <ul className="flex flex-col gap-0.5 text-slate-400">
                    {federated.privacy_log.map((line, i) => (
                      <li key={i}>› {line}</li>
                    ))}
                  </ul>
                </section>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
