# Deploying Nexus-Q

Split architecture — each half runs where it fits:

| Piece | Host | Why |
| --- | --- | --- |
| React terminal (frontend/) | **Vercel** | static build, CDN, instant deploys |
| Quantum API (backend/) | **Render / Railway / Fly.io** (Docker) | Qiskit stack needs ~600 MB of deps, 60 s+ CPU time, and the Elliptic CSVs on disk — none fit Vercel serverless |

---

## 1. Backend (Render — free tier works)

1. Push this repo to GitHub.
2. On <https://render.com>: **New → Web Service → connect the repo**.
   - **Root directory:** `backend`
   - **Runtime:** Docker (uses `backend/Dockerfile`)
   - **Instance type:** Free is fine for demos (cold starts ~30 s)
3. Environment variables:
   - `NEXUSQ_ALLOWED_ORIGINS=https://<your-vercel-url>` (add preview URLs too, comma-separated)
   - `GROQ_API_KEY`, `GROQ_API_KEY_FALLBACK` — the copilot keys
4. Deploy. Health check: `https://<service>.onrender.com/api/health`

### The Elliptic dataset on the backend

`backend/data/elliptic/` is **not** in the Docker image by default (690 MB).
Options:

- **Bake it** — uncomment the `COPY data/elliptic/` line in `backend/Dockerfile`
  (image becomes ~750 MB; fine on Render, may exceed Railway's default).
- **Mount a disk** — attach a 1 GB disk at `/srv/data` (Render/Railway support
  persistent disks) and upload the three CSVs once:
  `scp backend/data/elliptic/*.csv <host>:/srv/data/elliptic/`
- **Skip it** — without the CSVs, scans with `dataset: "elliptic"` gracefully
  fall back to the classical baseline; synthetic mode works fully.

## 2. Frontend (Vercel)

1. On <https://vercel.com>: **Add New → Project → import the repo**.
2. Configure:
   - **Root directory:** `frontend`
   - Framework preset: **Vite** (build `npm run build`, output `dist` — auto-detected)
3. Environment variable (Production + Preview):
   - `VITE_API_BASE = https://<your-render-service>.onrender.com`
4. Deploy. Done — `vercel.json` already handles SPA rewrites and asset caching.

## 3. Local dev unchanged

```powershell
.\start.bat          # backend :8000 + frontend :5173 (Vite proxies /api)
```

`VITE_API_BASE` is only needed on Vercel; locally the proxy handles it.

## Notes & limits

- **Free-tier cold starts:** Render free spins the API down after ~15 min idle;
  the first request then takes ~30 s (the UI just shows the scan-line longer).
- **Keep keys secret:** set Groq keys only in the Render dashboard, never in the
  repo. `frontend/.env` (if any) must stay out of git too.
- **CORS:** any new Vercel preview domain must be added to
  `NEXUSQ_ALLOWED_ORIGINS` or the browser will block API calls.
- Vercel alone *can* work if you gut the quantum parts and proxy to a hosted
  sandbox (e.g. IBM Quantum) — but then the demo depends on network latency to
  real hardware and loses the instant fallback behavior.
