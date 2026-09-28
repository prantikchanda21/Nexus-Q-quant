import { useEffect, useRef, useState } from "react";
import { sendChat } from "../api/client";
import brandIcon from "../assets/nexusq-icon.png";

interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  /** Provenance of assistant turns (rendered as a small badge). */
  meta?: { mode: string; key_label: string | null; model: string | null };
}

interface AssistantChatProps {
  open: boolean;
  onToggle: () => void;
}

const SUGGESTIONS = [
  "Explain the last contagion shock",
  "Why is systemic risk highest there?",
  "How does the federated AML scan protect privacy?",
  "What does coherence leakage mean?",
];

/**
 * Analyst copilot drawer. Every shock/scan the user runs is logged to the
 * backend assessment ledger; questions are grounded against that ledger via
 * Groq (primary key → fallback key → deterministic local briefing).
 */
export default function AssistantChat({ open, onToggle }: AssistantChatProps) {
  const [turns, setTurns] = useState<ChatTurn[]>([
    {
      role: "assistant",
      content:
        "Analyst copilot online. I can explain any assessment you run — contagion shocks, market waves, federated AML scans — with the exact parameters and results. Ask me anything.",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, open]);

  const ask = async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed || busy) return;

    const history = turns
      .filter((turn) => !turn.content.startsWith("Analyst copilot online"))
      .map((turn) => ({ role: turn.role, content: turn.content }));

    setTurns((prev) => [...prev, { role: "user", content: trimmed }]);
    setInput("");
    setBusy(true);
    try {
      const response = await sendChat({ message: trimmed, history });
      setTurns((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.answer,
          meta: {
            mode: response.mode,
            key_label: response.key_label,
            model: response.model,
          },
        },
      ]);
    } catch {
      setTurns((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "The assistant backend is unreachable. Is the FastAPI server running on :8000?",
        },
      ]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className={`panel pointer-events-auto absolute right-3 bottom-3 z-30 flex w-96 flex-col overflow-hidden transition-all duration-300 ${
        open ? "h-[30rem]" : "h-11"
      }`}
    >
      <header
        className="flex cursor-pointer select-none items-center justify-between border-b border-terminal-border px-3 py-2.5 text-xs uppercase tracking-widest text-sky-300/90"
        onClick={onToggle}
      >
        <span className="flex items-center gap-2">
          <img
            src={brandIcon}
            alt=""
            className="h-5 w-5 rounded object-contain drop-shadow-[0_0_8px_rgba(56,189,248,0.6)]"
          />
          analyst copilot
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
        </span>
        <span className="text-slate-500">{open ? "▾" : "▴"}</span>
      </header>

      {open && (
        <>
          <div ref={scrollRef} className="min-h-0 flex-1 space-y-2.5 overflow-y-auto px-3 py-3">
            {turns.map((turn, index) => (
              <div key={index} className={`flex ${turn.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-lg px-3 py-2 text-[11px] leading-relaxed ${
                    turn.role === "user"
                      ? "bg-sky-950/70 text-sky-100"
                      : "border border-terminal-border bg-black/40 text-slate-300"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{turn.content}</p>
                  {turn.meta && (
                    <p className="mt-1.5 flex items-center gap-1.5 text-[9px] uppercase tracking-wider text-slate-500">
                      <span
                        className={`rounded px-1 ${
                          turn.meta.mode === "groq"
                            ? "bg-emerald-950 text-emerald-400"
                            : "bg-amber-950 text-amber-400"
                        }`}
                      >
                        {turn.meta.mode === "groq" ? `groq · ${turn.meta.key_label}` : "local briefing"}
                      </span>
                      {turn.meta.model && <span>{turn.meta.model}</span>}
                    </p>
                  )}
                </div>
              </div>
            ))}
            {busy && (
              <div className="flex justify-start">
                <div className="rounded-lg border border-terminal-border bg-black/40 px-3 py-2 text-[11px] text-slate-500">
                  analyzing assessment ledger…
                </div>
              </div>
            )}
          </div>

          {turns.length <= 1 && (
            <div className="flex flex-wrap gap-1.5 px-3 pb-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => void ask(suggestion)}
                  className="rounded-full border border-terminal-border bg-black/30 px-2.5 py-1 text-[10px] text-slate-400 transition-colors hover:border-sky-700 hover:text-sky-300"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          <form
            className="flex gap-2 border-t border-terminal-border p-2"
            onSubmit={(event) => {
              event.preventDefault();
              void ask(input);
            }}
          >
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="ask about any assessment…"
              className="min-w-0 flex-1 rounded border border-terminal-border bg-black/50 px-2.5 py-1.5 text-[11px] text-slate-200 placeholder:text-slate-600 focus:border-sky-700 focus:outline-none"
            />
            <button
              type="submit"
              disabled={busy || !input.trim()}
              className="btn border-sky-600/60 bg-sky-950/50 px-3 py-1.5 text-[10px] text-sky-300 hover:bg-sky-900/50"
            >
              send
            </button>
          </form>
        </>
      )}
    </div>
  );
}
