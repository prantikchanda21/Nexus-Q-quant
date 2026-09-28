<div align="center">
  <img src="assets_nexusq_readme.png" alt="Nexus-Q — Quantum Financial Terminal" width="560" />
</div>

# NEXUS-Q

**Quantum-Enhanced Financial Contagion & AML Graph Triage Platform**
_Q-HACK India 2026 / Qiskit Fall Fest Hackathon_

Nexus-Q is a hybrid quantum-classical financial terminal that solves three systemic-risk
problems with Qiskit 1.x Primitives V2:

1. **Dynamic Financial Contagion** — cascading bank defaults modeled as a
   Continuous-Time Quantum Walk (CTQW) over the interbank exposure network.
2. **Privacy-Preserving AML Triage** — obfuscated circular money flows flagged by a
   Quantum Support Vector Classifier trained via Quantum Federated Learning (QFL):
   bank silos exchange only analytic gradients, never raw transactions.
3. **Live-Market Contagion (yfinance)** — a header toggle flips the whole terminal to
   real market data: tickers become nodes, return correlations become exposure
   corridors, and the *same* CTQW engine propagates distress from a shocked stock
   into correlated stocks and the funds tracking them.

---

## 1 · System architecture

Every request flows from the React terminal through the Vite dev proxy into FastAPI,
which routes it to one of three quantum engines:

```
                        ┌─────────────────────────────────────────────┐
                        │              BROWSER TERMINAL               │
                        │         React 19 · Vite · Tailwind          │
                        │                                             │
                        │  Network3DCanvas   three.js + 3d-force-graph│
                        │  ShockControlPanel shock sliders + ranking  │
                        │  QuantumTelemetry  histogram · QASM3 · NISQ │
                        │  AssistantChat     Groq-grounded copilot    │
                        └──────────────────────┬──────────────────────┘
                                               │  fetch  (VITE_API_BASE
                                               │         or Vite /api proxy)
                                               ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                    FASTAPI BACKEND  ·  :8000                     │
        │                    app/main.py · CORS origins.py                 │
        │                                                                  │
        │   /api/contagion/shock ──┐                                       │
        │   /api/market/*  ────────┼──► [assessment_log] ──► /api/chat     │
        │   /api/aml/federated-scan┘     (ledger digest grounds copilot)   │
        └───────┬───────────────────────┬───────────────────────┬──────────┘
                │                       │                       │
                ▼                       ▼                       ▼
   ╔═══════════════════════╗ ╔══════════════════════╗ ╔═══════════════════════╗
   ║   CTQW CONTAGION      ║ ║   QFL FEDERATED      ║ ║   GROQ COPILOT        ║
   ║   contagion_ctqw.py   ║ ║   aml_federated_qsvc ║ ║   groq_client.py      ║
   ║                       ║ ║                      ║ ║                       ║
   ║  PauliEvolutionGate   ║ ║  ZZFeatureMap +      ║ ║  gpt-oss-120b         ║
   ║  StatevectorSampler   ║ ║  RealAmplitudes QSVC ║ ║  └─ fallback 20b      ║
   ║  NISQ resilience path ║ ║  ParamShiftGradient  ║ ║  └─ fallback local    ║
   ╚═════════╤═════════════╝ ╚═════════╤════════════╝ ╚═══════════════════════╝
             │                         │
             ▼                         ▼
   ┌───────────────────┐     ┌──────────────────────────────────┐
   │ graph_generator   │     │  elliptic_data.py                │
   │ (Eisenberg–Noe    │     │  203,770 txs · 4,545 illicit     │
   │  synthetic banks) │     │  234,356 edges · PCA stream      │
   └───────────────────┘     └──────────────────────────────────┘
   ┌───────────────────┐
   │ market_network    │  yfinance correlations → exposure matrix
   │ (TTL 15 min cache)│  degraded:true on Yahoo failure
   └───────────────────┘

   every shock / scan ──► record_assessment() ──► build_context_digest()
                                                  (6,000-char budget) ──► /api/chat
```

### Repository map

```
nexus-q/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI entrypoint, CORS, routes
│   │   └── core/
│   │       ├── graph_generator.py      # NetworkX interbank graph + k-hop slicer
│   │       ├── contagion_ctqw.py       # CTQW engine (PauliEvolutionGate + SamplerV2)
│   │       ├── aml_federated_qsvc.py   # QFL loop (ZZFeatureMap + ParamShift gradients)
│   │       ├── elliptic_data.py        # Elliptic Bitcoin loader (streamed PCA cache)
│   │       ├── market_network.py       # yfinance correlation → exposure network
│   │       ├── assessment_log.py       # in-memory assessment ledger (deque, 50)
│   │       ├── groq_client.py          # two-key copilot chain + local fallback
│   │       └── origins.py              # NEXUSQ_ALLOWED_ORIGINS CORS
│   ├── data/elliptic/                  # Weber et al. (2019) CSVs (~690 MB)
│   ├── smoke_test.py / sweep_banks.py / market_test.py
│   └── Dockerfile · requirements.txt
├── frontend/
│   └── src/  (Network3DCanvas · ShockControlPanel · QuantumTelemetry
│              AssistantChat · api/client.ts · types/nexusq.ts · App.tsx)
├── demo_build/                         # narrated demo video pipeline
├── DEPLOY.md · render.yaml · start.bat / start.sh
└── README.md
```

---

## 2 · Engine I — CTQW contagion (`contagion_ctqw.py`)

### Data pipeline

```
  Eisenberg–Noe synthetic bank network        (graph_generator.py, seed 42)
  ┌─────────────────────────────────────┐
  │ 24 banks · 63 exposure corridors    │      a shock at bank k selects a
  │ L[i][j] = bilateral exposure        │ ───► k-hop slice:  4 ≤ n ≤ 10 banks
  │ c[i]    = equity capital            │      (the qubit budget — one bank
  └─────────────────────────────────────┘       = one qubit)
                 │
                 ▼
  ┌───────────────────────────────────────────────────────────┐
  │  symmetrize_exposure_matrix():   A_sym = (A + Aᵀ) / 2     │
  └───────────────────────────────────────────────────────────┘
                 │
                 ▼
  ┌───────────────────────────────────────────────────────────┐
  │  build_hamiltonian():   H = γ · A_sym                     │
  │                         γ = liquidity velocity (slider)   │
  │                         → SparsePauliOp                   │
  └───────────────────────────────────────────────────────────┘
```

### The quantum core

Because `A_sym` is Hermitian with **zero diagonal**, each entry `w_ij` decomposes
**exactly** — no Jordan–Wigner strings, no ancillas — into an XX + YY hopping pair
on qubits `i, j`:

```
        (X_i X_j + Y_i Y_j) / 2  =  |01⟩⟨10| + |10⟩⟨01|

        ── the exact single-excitation hop operator ──

        H = γ · Σ_(i<j)  (w_ij / 2) · (X_i X_j + Y_i Y_j)

        |ψ(0)⟩ = |shocked_bank⟩          ← a single excitation on bank k

        U(t) = exp(−i H t)               ← continuous-time quantum walk
```

This preserves the one-excitation subspace (up to Lie–Trotter error), so the walk
amplitude on bank `i` is precisely the probability that distress has propagated
there.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ build_circuit():                                                │
  │                                                                 │
  │        ┌───┐  ╔══════════════════════════╗                      │
  │   q_0  ┤ X ┤──╢                          ╟── M ──► bitstring    │
  │        └───┘  ║  PauliEvolutionGate      ╟── M                  │
  │   q_1 ────────╢  (H, t, reps = 4)        ╟── M      qiskit      │
  │        ...    ║  Lie–Trotter synthesis   ╟── M  Statevector-    │
  │   q_n ────────╢                          ╟── M  Sampler (V2)    │
  │               ╚══════════════════════════╝                      │
  └─────────────────────────────────────────────────────────────────┘
                 │  shots
                 ▼
  failure probability p_i  =  Σ over bitstrings with qubit i = 1
  coherence_leakage        =  1 − Σ_i p_i      (Trotter / noise gauge)
  _synthesize_qasm()       =  OpenQASM 3 export for the telemetry drawer
```

**Hardware path.** If an IBM `BackendV2` is supplied, `_run_quantum` switches to
`qiskit_ibm_runtime.SamplerV2` under `NISQResilienceConfig`:

```
  resilience_level = 1      dynamical_decoupling = True
  gate_twirling    = True   optimization_level   = 3
  + ISA transpilation for the target coupling map
```

**Classical fallback.** Any quantum-path failure (no Qiskit, bad slice, hardware
error) degrades to an Eisenberg–Noe loss cascade blended with PageRank
(`degree_centrality_cascade`, `page_rank_cascade`), annotated
`mode: "classical_fallback"` — the terminal never goes blank.

**Verified numbers:** default 5-qubit shock ≈ 30–40 ms (budget: 1.5 s NISQ realtime),
leakage 0.0000, sweep of all 24 banks peaks at **BANK-019 spill = 1.000**.

### Live-market mode (`market_network.py`)

The same engine, different network builder:

```
   yfinance close prices ──► return series r_i(t)
                 │
                 ▼
   ρ_ij = corr(r_i, r_j)          edge weight  =  max(0, ρ_ij)
                 │
                 ▼
   node = ticker · equity capital ∝ 1/σ_annualized
   universes: nifty50 · it · banks (SBIN.NS …) · mixedfunds
   slice keeps 10 highest-vol assets (qubit budget) · TTL cache 15 min
   Yahoo unreachable → deterministic sector-block network, degraded:true
```

Verified: shock SBIN.NS → ICICIBANK propagation amplitude 0.214 in the banks universe.

---

## 3 · Engine II — Federated QSVC for AML (`aml_federated_qsvc.py`)

### Qubit circuit per transaction

```
     x = (x₁, x₂, x₃, x₄)  normalized PCA / graph features

        φ(x) = x · π          (scaled into [0, π] — _bind_sample)

     ┌────┐┌──────────────┐┌────┐┌────┐┌──────────────┐┌────┐┌────┐
  q₀ ┤ H  ├┤ Rz(φ₁)       ├┤ √X ├┤ H  ├┤ Rz(φ₁φ₂)+…   ├┤ √X ├┤ H  ├┤ Ry(θ) ├── ⟨Z₀⟩
     └────┘└──────────────┘└────┘└────┘└──────────────┘└────┘└────┘└────────┘
        ZZFeatureMap(4, reps=2)      entangling ZZ ring      RealAmplitudes
        (feature encoding)                                  (circular, trainable θ)
```

Observable is `⟨Z₀⟩` on qubit 0, evaluated analytically by the V2
`StatevectorEstimator` with batched PUBs `(circuit, Z0, theta)`.

### Training: parameter-shift gradients

Per-sample squared-hinge-free MSE loss `L_i = (⟨Z₀⟩_i − y_i)²`. The chain rule with
`ParamShiftEstimatorGradient` (exact, shift ±π/2):

```
   ∂L_i/∂θ_j  =  2 · (⟨Z₀⟩_i − y_i) · ∂⟨Z₀⟩_i/∂θ_j

   ┌───────────────────────────────────────────────────────────────┐
   │  StatevectorEstimator (V2 PUBs)                               │
   │        │                                                      │
   │        ▼                                                      │
   │  _V2EstimatorLegacyBridge  ──► ParamShiftEstimatorGradient    │
   │        (the PUB bridge is load-bearing — never delete it)     │
   └───────────────────────────────────────────────────────────────┘
```

### Federation: FedSGD / FedAvg

Three silos train locally; **only gradients and scores cross silo boundaries**:

```
   ┌── SILO A ──┐   ┌── SILO B ──┐   ┌── SILO C ──┐
   │ raw ledger │   │ raw ledger │   │ raw ledger │
   │ local ∇L_A │   │ local ∇L_B │   │ local ∇L_C │        12-float gradients
   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘        only — no raw txs
         │                │                │
         ▼                ▼                ▼
   ┌─────────────────────────────────────────────┐
   │  SERVER  global_update():                   │
   │                                             │
   │  θ ← θ − η · (1/C) Σ_c ∂L_c/∂θ              │   FedSGD aggregation
   │  class-weighted loss (illicit ≈ 10× rarer)  │
   └─────────────────────────────────────────────┘
         │
         ▼   federated validation = weighted mean of client
             acc / precision / recall / F1 · FedSGD loss curve
             (synthetic: 1.019 → 0.826 over 8 rounds, acc ≈ 0.71)
```

Every egress is recorded in the privacy audit log —
`"0 raw transaction records egressed"`.

### Real data — the Elliptic Bitcoin ledger (`elliptic_data.py`)

Select **"Elliptic · real Bitcoin"** in the AML panel to train on Weber et al. (2019):

```
   elliptic_txs_classes.csv   class "1" illicit · "2" licit
   elliptic_txs_edgelist.csv  payment edges
   elliptic_txs_features.csv  166-dim PCA (streamed — 690 MB parsed once)

   ┌────────────────────────────────────────────────────────────┐
   │ _stream_all_pca()  parses ALL tx PCA comps 0–3 once,       │
   │ cached in _FEATURES_CACHE keyed by (path, mtime)           │
   │   cold scan 78 s  →  warm ≈ 21 s  (training itself)        │
   ├────────────────────────────────────────────────────────────┤
   │ load_elliptic_sample(n_clients, …) stratified illicit/licit│
   │ pools, _SPLIT_CACHE keyed by params · silos ELL-SILO-XX    │
   ├────────────────────────────────────────────────────────────┤
   │ velocity() = 2-hop BFS capped · feature_source = "pca"     │
   │ if ≥ 80 % coverage else "graph"                            │
   └────────────────────────────────────────────────────────────┘

   203,770 transactions · 4,545 illicit · 42,019 licit · 234,356 edges
   Elliptic accuracy ≈ 0.67–0.73 (η = 8.0, 12–16 rounds); F1 ≈ 0.2–0.5 —
   honest 4-qubit variance on ~90 sampled transactions.
```

Anomaly scoring adds `fidelity_entropy = −F·lnF −(1−F)·ln(1−F)` against the
anomalous-prototype state, highlighting transactions straddling the
legitimate/obfuscated boundary.

---

## 4 · Analyst copilot (`assessment_log.py` + `groq_client.py`)

```
   shock / scan ──► record_assessment(kind, summary, details) ──► seq id
                          (thread-safe deque, maxlen = 50)
                                      │
   POST /api/chat ────────────────────┤
                                      ▼
                     build_context_digest(limit)   ≤ 6,000 chars
                                      │
                                      ▼
   ┌────────────────── Groq fallback chain ──────────────────┐
   │  GROQ_API_KEY ──► openai/gpt-oss-120b                   │
   │      │ 401 / 429 / 5xx                                  │
   │      ▼                                                  │
   │  GROQ_API_KEY_FALLBACK ──► openai/gpt-oss-20b           │
   │      │                                                  │
   │      ▼                                                  │
   │  deterministic local summarizer  (mode: "local")        │
   └─────────────────────────────────────────────────────────┘

   answers cite run numbers, parameters and outcomes verbatim from the ledger.
```

---

## 5 · API reference

| Route | Method | Payload → Response |
| --- | --- | --- |
| `/api/health` | GET | service + engine inventory |
| `/api/graph/summary` | GET | full interbank topology for the 3D canvas |
| `/api/contagion/shock` | POST | `{shocked_node_id, time_t, gamma?, hops?, shots?}` → blast radius, histogram, QASM3, telemetry |
| `/api/market/graph?universe=&window=` | GET | live correlation network (yfinance, TTL-cached, synthetic fallback) |
| `/api/market/contagion/shock` | POST | `{shocked_ticker, time_t, universe, window, gamma?, shots?}` → market blast radius |
| `/api/aml/federated-scan` | POST | `{n_rounds 1–12, learning_rate ≤10, dataset: synthetic\|elliptic}` → federated metrics, anomaly scores, privacy log |
| `/api/chat` | POST | `{message, history?}` → copilot answer (`mode`/`key_label`/`model`) |

### Market-mode concept mapping

| Interbank concept | Live-market analogue |
| --- | --- |
| Bank | Stock / ETF / mutual-fund proxy ticker |
| Bilateral exposure L[i][j] | return correlation ρ_ij (exposure = max(0, ρ)²) |
| Equity capital | 1 / σ_annualized (low-vol assets are "solvent") |
| Shocked bank defaults | Shocked asset enters a drawdown regime |
| Mutual funds | Proxy high-beta slice of disclosed holdings |

---

## 6 · Run it

```bash
# backend (Python ≥ 3.10)
cd backend
python -m venv ../.venv
../.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: ../.venv/bin/pip
../.venv/Scripts/python smoke_test.py                 # engine smoke test
../.venv/Scripts/python -m uvicorn app.main:app --port 8000

# frontend (Node ≥ 20)
cd frontend
npm install
npm run dev            # http://localhost:5173, /api proxied to :8000
```

Windows one-click: `.\start.bat` (backend :8000 + frontend :5173).

Copilot keys: `cp backend/.env.example backend/.env`, add free keys from
<https://console.groq.com/keys>. Without keys everything still works — the chatbox
answers from the deterministic local summarizer.

### Demo budget

Graph slices are clamped to **4–10 qubits**; the default 5-qubit shock runs in
**≈ 40 ms** (well under the 1.5 s NISQ-realtime budget) and the 3-client × 8-round
federated scan trains in **≈ 9 s** with a visible FedSGD loss curve (1.019 → 0.826)
reaching ≈ 0.71 federated accuracy on the synthetic silos.

## 7 · Deploy

Split architecture — each half runs where it fits (full guide in `DEPLOY.md`):

| Piece | Host | Why |
| --- | --- | --- |
| React terminal (`frontend/`) | **Vercel** (`vercel.json` SPA rewrites) | static build, CDN |
| Quantum API (`backend/`) | **Render / Railway / Fly.io** (Docker) | Qiskit stack + Elliptic CSVs don't fit serverless |

Backend env: `NEXUSQ_ALLOWED_ORIGINS`, `GROQ_API_KEY`, `GROQ_API_KEY_FALLBACK`;
frontend env: `VITE_API_BASE`. The 690 MB Elliptic CSVs are optionally baked into
the Docker image or mounted as a persistent disk; without them elliptic scans
gracefully fall back to the classical baseline.

with live telemetry, Elliptic federated scan with metric strip, market-mode SBIN
shock, and a grounded copilot answer.
