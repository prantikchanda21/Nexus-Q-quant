"""Live-market contagion network built from yfinance data.

Maps the Nexus-Q interbank contagion model onto real market structure:

    node  = a stock, ETF, or mutual fund (ticker)
    edge  = return correlation between two assets (rolling window)
    weight = max(0, ρ_ij)  →  "correlation exposure" (contagion channel)

High correlation is the market analogue of a bilateral lending corridor:
distress (drawdown) propagates along it. The resulting weighted adjacency
feeds the *same* CTQW engine used for interbank contagion — H = γA_sym,
|ψ(0)⟩ = |shocked asset⟩, U(t) = e^{-iHt} — so a shock on a large-cap stock
shows the distress wave flowing into correlated stocks and into the mutual
funds / ETFs that track the same underlying assets.

Networks are cached in-process with a TTL so repeated shocks do not re-download
data, and every failure path degrades to a deterministic synthetic correlation
network (annotated in the payload) so the demo never breaks on rate limits.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]
    YFINANCE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Universe definitions — NSE-listed equities, ETFs, and mutual-fund proxies.
# Mutual funds have no intraday ticker on Yahoo; we proxy each fund by the
# high-beta slice of its disclosed portfolio (standard practice for NAV
# attribution), so "shock the fund" ≈ "shock its top holdings".
# ---------------------------------------------------------------------------
MUTUAL_FUND_PROXIES: Dict[str, List[str]] = {
    "PARAG-PARIKH-CF": ["TCS.NS", "HDFCBANK.NS", "ITC.NS", "TITAN.NS", "MPHASIS.NS"],
    "SBI-BLUECHIP": ["ICICIBANK.NS", "INFY.NS", "SBIN.NS", "LT.NS", "TATASTEEL.NS"],
    "QUANT-SMALLCAP": ["IRFC.NS", "RVNL.NS", "IEX.NS", "SUZLON.NS", "JIOFIN.NS"],
}

UNIVERSES: Dict[str, Dict[str, object]] = {
    "nifty50": {
        "label": "NIFTY 50 · stocks + ETFs",
        "tickers": [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
            "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
            "BAJFINANCE.NS", "MARUTI.NS", "TITAN.NS", "TATASTEEL.NS", "WIPRO.NS",
            "AXISBANK.NS", "SUNPHARMA.NS", "ONGC.NS", "ADANIENT.NS", "DMART.NS",
            "^NSEI",        # Nifty 50 index
            "^CNXIT",       # Nifty IT sector index
            "GOLDBEES.NS",  # Gold ETF
            "NIFTYBEES.NS", # Nifty index ETF
        ],
        "sector_of": {},
    },
    "it": {
        "label": "IT pack · TCS / INFY / WIPRO + funds",
        "tickers": [
            "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
            "^CNXIT", "NIFTYBEES.NS",
        ],
        "sector_of": {},
    },
    "banks": {
        "label": "Banks & financials + sector ETFs",
        "tickers": [
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
            "BAJFINANCE.NS", "^NSEBANK", "GOLDBEES.NS",
        ],
        "sector_of": {},
    },
    "mixedfunds": {
        "label": "Stocks + mutual-fund proxies",
        "tickers": [
            "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS", "TITAN.NS",
            "MPHASIS.NS", "INFY.NS", "SBIN.NS", "LT.NS",
            "IRFC.NS", "RVNL.NS", "SUZLON.NS",
        ],
        "sector_of": {},
    },
}


class MarketDataError(RuntimeError):
    """Raised when live data cannot be fetched at all."""


@dataclass
class MarketNetwork:
    """A correlation-contagion network over market tickers.

    Attributes:
        universe: Universe key used to build the network.
        node_ids: Original ticker indices, index-aligned with the matrix.
        matrix: Weighted adjacency; matrix[i][j] = max(0, ρ_ij) exposure.
        capitals: "Capital" per asset = 1/annualized volatility (defensive
            assets are more solvent: they absorb shock without defaulting).
        labels: Display labels keyed by ticker.
        sectors: Sector classification keyed by ticker (heuristic).
        window: Correlation window in trading days.
        data_age_s: Age of the underlying price data (fetch latency).
        degraded: True when the synthetic fallback network was used.
        fallback_reason: Why the fallback fired (if any).
        stats: Headline topology statistics.
    """

    universe: str
    node_ids: List[str]
    matrix: List[List[float]]
    capitals: List[float]
    labels: Dict[str, str] = field(default_factory=dict)
    sectors: Dict[str, str] = field(default_factory=dict)
    window: int = 60
    data_age_s: float = 0.0
    degraded: bool = False
    fallback_reason: Optional[str] = None
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def n_assets(self) -> int:
        return len(self.node_ids)

    def edge_rows(self) -> List[Dict[str, object]]:
        """Edges for the /api/market/graph payload (source/target tickers)."""
        edges: List[Dict[str, object]] = []
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                weight = self.matrix[i][j]
                if weight > 0.02:
                    edges.append(
                        {
                            "source": self.node_ids[i],
                            "target": self.node_ids[j],
                            "exposure": round(weight, 4),
                        }
                    )
        return edges


def _classify_sector(ticker: str) -> str:
    """Heuristic sector classifier for coloring + grouping in the UI."""
    t = ticker.upper()
    if t.startswith("^"):
        return "index"
    if "BEES" in t or "ETF" in t or t.endswith(".BO"):
        return "etf"
    banks = ("HDFC", "ICICI", "SBIN", "AXIS", "KOTAK", "BAJFIN", "JIOFIN", "IRFC")
    it = ("TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "MPHASIS", "PERSISTENT")
    metal = ("TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL")
    energy = ("RELIANCE", "ONGC", "NTPC", "POWERGRID", "ADANIENT", "SUZLON")
    if any(k in t for k in banks):
        return "bank"
    if any(k in t for k in it):
        return "it"
    if any(k in t for k in metal):
        return "metal"
    if any(k in t for k in energy):
        return "energy"
    if "MARUTI" in t or "TITAN" in t or "DMART" in t:
        return "consumer"
    return "stock"


# ---------------------------------------------------------------------------
# Data fetch & correlation → exposure graph
# ---------------------------------------------------------------------------
def _fetch_returns(tickers: List[str], window: int) -> Tuple[np.ndarray, List[str], float]:
    """Download adjusted closes and return (returns_matrix, ok_tickers, latency).

    Assets with too little data are dropped rather than failing the whole
    network; the caller keeps only tickers that survived.
    """
    import pandas as pd

    started = time.perf_counter()
    data = yf.download(
        tickers,
        period=f"{max(window * 3, 120)}d",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    latency = time.perf_counter() - started

    multi_index = isinstance(data.columns, pd.MultiIndex)
    closes: Dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            series = (data[ticker]["Close"] if multi_index else data["Close"]).dropna()
        except Exception:  # noqa: BLE001 — one bad ticker must not kill the build
            continue
        if len(series) >= window + 5:
            closes[ticker] = series

    if len(closes) < 4:
        raise MarketDataError(f"only {len(closes)} tickers returned usable data")

    frame = pd.DataFrame(closes)
    returns = frame.pct_change().dropna().iloc[-window:]
    return returns.to_numpy(dtype=float), list(returns.columns), latency


def _correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """Pearson correlation across assets (columns)."""
    return np.corrcoef(returns, rowvar=False)


def _synthetic_correlation(tickers: List[str], seed: int = 7) -> np.ndarray:
    """Deterministic fallback correlation network (sector-block structure).

    Used only when yfinance is unreachable (rate limit, offline demo). Assets
    in the same heuristic sector get ρ ~ 0.55, the market factor adds a
    baseline ρ ~ 0.25 everywhere.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    corr = np.full((n, n), 0.25)
    for i in range(n):
        corr[i, i] = 1.0
        for j in range(i + 1, n):
            if _classify_sector(tickers[i]) == _classify_sector(tickers[j]):
                corr[i, j] = corr[j, i] = 0.55
    noise = rng.normal(0.0, 0.05, size=(n, n))
    corr = np.clip(corr + noise, -1.0, 1.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


_CACHE: Dict[Tuple[str, int], Tuple[float, MarketNetwork]] = {}
CACHE_TTL_S = 900.0


def build_market_network(
    universe: str = "nifty50",
    window: int = 60,
    max_assets: int = 10,
    seed: int = 42,
) -> MarketNetwork:
    """Build (or fetch from TTL cache) a correlation-contagion network.

    Args:
        universe: Key into UNIVERSES.
        window: Rolling correlation window in trading days.
        max_assets: Slice cap — the CTQW qubit budget (4–10).
        seed: Deterministic seed for the fallback path.

    Returns:
        A :class:`MarketNetwork` with a symmetric exposure matrix in [0, 1].
    """
    cache_key = (universe, window)
    cached = _CACHE.get(cache_key)
    if cached and time.time() - cached[0] < CACHE_TTL_S:
        return cached[1]

    spec = UNIVERSES.get(universe)
    if spec is None:
        raise MarketDataError(f"unknown universe '{universe}'")
    tickers: List[str] = list(spec["tickers"])  # type: ignore[arg-type]

    degraded = False
    fallback_reason: Optional[str] = None
    latency = 0.0

    if YFINANCE_AVAILABLE:
        try:
            returns, survived, latency = _fetch_returns(tickers, window)
            corr = _correlation_matrix(returns)
            tickers = survived
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the demo
            LOGGER.warning("yfinance fetch failed (%s); using synthetic network", exc)
            corr = _synthetic_correlation(tickers, seed=seed)
            degraded = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        corr = _synthetic_correlation(tickers, seed=seed)
        degraded = True
        fallback_reason = "yfinance not installed"

    # --- Rank assets by volatility: keep the demo slice liquid + diverse ----
    # Capital = 1/σ (defensive assets are "solvent"); we rank by σ descending
    # so the sliced network is full of real contagion channels.
    vols: Optional[np.ndarray] = None if degraded else np.std(returns, axis=0, ddof=1)
    if vols is not None:
        order = sorted(range(len(tickers)), key=lambda i: -float(vols[i]))
    else:
        order = list(range(len(tickers)))

    keep = order[:max_assets]
    # Re-sort kept tickers alphabetically so slices are stable across calls.
    keep = sorted(keep, key=lambda i: tickers[i])
    slice_tickers = [tickers[i] for i in keep]

    n = len(slice_tickers)
    index_of = {ticker: i for i, ticker in enumerate(slice_tickers)}
    matrix = [[0.0] * n for _ in range(n)]
    capitals: List[float] = []
    labels: Dict[str, str] = {}
    sectors: Dict[str, str] = {}

    for i, ticker_i in enumerate(slice_tickers):
        labels[ticker_i] = ticker_i.replace(".NS", "")
        sectors[ticker_i] = _classify_sector(ticker_i)
        # Capital = 1/σ_annualized: low-vol assets (GOLDBEES) are "solvent",
        # high-vol smallcaps burn capital fast under shock.
        sigma = float(vols[keep[i]]) if vols is not None else 0.015
        sigma_annual = sigma * math.sqrt(252.0)
        capitals.append(round(min(5.0, max(0.2, 1.0 / max(sigma_annual, 1e-4))), 3))
        for j, ticker_j in enumerate(slice_tickers):
            if i == j:
                continue
            rho = float(corr[keep[i], keep[j]])
            # Asymmetric exposure: ρ² weights genuine co-movement over noise,
            # and the corridor only carries positive correlation.
            matrix[i][j] = round(max(0.0, rho) ** 2, 4)

    # Topology stats for the header chips.
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > 0:
                graph.add_edge(i, j, weight=matrix[i][j])

    network = MarketNetwork(
        universe=universe,
        node_ids=slice_tickers,
        matrix=matrix,
        capitals=capitals,
        labels=labels,
        sectors=sectors,
        window=window,
        data_age_s=round(latency, 3),
        degraded=degraded,
        fallback_reason=fallback_reason,
        stats={
            "n_assets": float(n),
            "n_edges": float(graph.number_of_edges()),
            "density": round(nx.density(graph), 4),
            "avg_corr": round(
                float(np.mean([matrix[i][j] for i in range(n) for j in range(n) if i != j])), 4
            ),
        },
    )
    _CACHE[cache_key] = (time.time(), network)
    return network
