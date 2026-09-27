"""Synthetic interbank exposure graph generation and slicing.

Builds Eisenberg–Noe style interbank lending networks with NetworkX and slices
them down to demo-sized subgraphs (4–10 nodes) so every quantum circuit fits
within the real-time simulation budget of the hackathon demo.

All quantities are in normalized notional units. Edge weights are bilateral
exposures ``L[i][j]`` = exposure of bank i to bank j (i lent to j).
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

# Demo budget: the CTQW Hamiltonian is mapped one Pauli term per edge, so we
# hard-cap the active slice to keep simulation under the ~1.5 s budget.
MIN_SLICE_NODES = 4
MAX_SLICE_NODES = 10


class GraphSizeError(ValueError):
    """Raised when a requested slice cannot fit the qubit budget."""


@dataclass
class InterbankSlice:
    """A demo-sized slice of the interbank network ready for quantization.

    Attributes:
        node_ids: Original bank IDs in the slice, index-aligned with the matrix.
        matrix: Weighted adjacency matrix; ``matrix[i][j]`` is exposure of i to j.
        capitals: Equity capital of each bank (Eisenberg–Noe solvency buffer).
        labels: Human-readable bank names keyed by original node id.
        edge_list: (source_local_idx, target_local_idx, exposure) triples.
    """

    node_ids: List[int]
    matrix: List[List[float]]
    capitals: List[float]
    labels: Dict[int, str]
    edge_list: List[Tuple[int, int, float]] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def total_exposure(self, node_local_idx: int) -> float:
        """Total bilateral exposure of a bank within the slice."""
        return sum(self.matrix[node_local_idx])


def _stable_rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_interbank_graph(
    n_banks: int = 24,
    seed: int = 42,
    density: float = 0.18,
) -> nx.DiGraph:
    """Generate a synthetic Eisenberg–Noe interbank lending network.

    Uses a Watts–Strogatz small-world topology (ring lattice + rewiring), which
    reproduces the clustered, core-periphery-ish structure observed in real
    interbank markets. Exposures are drawn from a log-normal distribution and
    normalized so each bank's total interbank assets are comparable.

    Args:
        n_banks: Number of banks in the full network (sliced later).
        seed: Deterministic RNG seed.
        density: Edge rewiring probability for the Watts–Strogatz graph.

    Returns:
        A directed weighted NetworkX graph. Edge attribute ``exposure`` holds
        the bilateral notional; node attribute ``capital`` holds equity.
    """
    if n_banks < MIN_SLICE_NODES:
        raise GraphSizeError(f"Need at least {MIN_SLICE_NODES} banks, got {n_banks}")

    rng = _stable_rng(seed)
    # Watts–Strogatz: k ≈ 4 neighbours on the ring keeps the graph sparse and
    # clustered like real interbank lending corridors.
    undirected = nx.watts_strogatz_graph(n_banks, k=4, p=density, seed=seed)

    graph = nx.DiGraph()
    for i in range(n_banks):
        capital = round(rng.uniform(8.0, 20.0), 2)
        graph.add_node(i, capital=capital, label=f"BANK-{i:03d}")

    for u, v in undirected.edges():
        # Randomly orient the lending direction; exposures are asymmetric.
        if rng.random() < 0.5:
            u, v = v, u
        exposure = round(rng.lognormvariate(mu=0.4, sigma=0.6), 3)
        graph.add_edge(u, v, exposure=exposure)
        # Add a smaller reverse exposure with some probability (corridor both ways).
        if rng.random() < 0.35 and not graph.has_edge(v, u):
            graph.add_edge(v, u, exposure=round(exposure * rng.uniform(0.2, 0.8), 3))

    # Guarantee every node has at least one in- or out-edge so no isolated
    # banks produce dead qubits in the Hamiltonian.
    for node in graph.nodes():
        if graph.degree(node) == 0:
            partner = rng.choice([n for n in graph.nodes() if n != node])
            graph.add_edge(node, partner, exposure=round(rng.uniform(0.5, 3.0), 3))

    return graph


def slice_around_node(
    graph: nx.DiGraph,
    center_node: int,
    hops: int = 1,
    max_nodes: int = MAX_SLICE_NODES,
) -> InterbankSlice:
    """Extract a k-hop neighbourhood around a shocked bank, capped in size.

    The slice is grown breadth-first from ``center_node`` so the most systemically
    relevant (closest) counterparties are kept when the neighbourhood exceeds the
    qubit budget.

    Args:
        graph: Full interbank network.
        center_node: The shocked bank (becomes qubit 0 / |0...0> seed state).
        hops: Number of hops to expand around the center.
        max_nodes: Hard cap on slice size (qubit budget).

    Returns:
        An :class:`InterbankSlice` with 4 <= n <= 10 nodes.

    Raises:
        GraphSizeError: If the slice cannot satisfy the qubit budget.
    """
    if center_node not in graph:
        raise GraphSizeError(f"Unknown node {center_node}")

    # BFS over the undirected projection to capture lenders and borrowers alike.
    undirected = graph.to_undirected()
    visited: Set[int] = {center_node}
    frontier: List[int] = [center_node]

    for _ in range(max(1, hops)):
        next_frontier: List[int] = []
        for node in frontier:
            for nb in sorted(undirected.neighbors(node)):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
        if len(visited) >= max_nodes:
            break

    ordered = sorted(visited, key=lambda n: (n != center_node, n))
    if len(ordered) > max_nodes:
        # Keep the center plus its closest counterparties (stable BFS order).
        keep = [center_node] + [
            n for n in ordered if n != center_node
        ][: max_nodes - 1]
        ordered = sorted(keep, key=lambda n: (n != center_node, n))

    if len(ordered) < MIN_SLICE_NODES:
        # Grow with globally nearest nodes (deterministic tie-break by id).
        for candidate in sorted(graph.nodes()):
            if candidate in ordered:
                continue
            ordered.append(candidate)
            if len(ordered) >= MIN_SLICE_NODES:
                break
        if len(ordered) < MIN_SLICE_NODES:
            raise GraphSizeError(
                "Slice too small to quantize; graph appears disconnected"
            )

    node_ids = ordered[:max_nodes]
    index_of = {node: i for i, node in enumerate(node_ids)}

    matrix = [[0.0] * len(node_ids) for _ in node_ids]
    capitals: List[float] = []
    labels: Dict[int, str] = {}
    edge_list: List[Tuple[int, int, float]] = []

    for i, node in enumerate(node_ids):
        capitals.append(float(graph.nodes[node].get("capital", 10.0)))
        labels[node] = str(graph.nodes[node].get("label", f"BANK-{node:03d}"))
        for j, other in enumerate(node_ids):
            if graph.has_edge(node, other):
                w = float(graph.edges[node, other].get("exposure", 1.0))
                matrix[i][j] = w
                edge_list.append((i, j, w))

    return InterbankSlice(
        node_ids=node_ids,
        matrix=matrix,
        capitals=capitals,
        labels=labels,
        edge_list=edge_list,
    )


def degree_centrality_cascade(
    matrix: Sequence[Sequence[float]],
    capitals: Sequence[float],
    shocked_local_idx: int,
    loss_propagation: float = 0.85,
) -> List[float]:
    """Classical Eisenberg–Noe style cascade baseline (fallback path).

    Iteratively propagates default losses through the exposure matrix: when a
    bank defaults, each counterparty absorbs a fraction of its interbank assets
    as a loss. Runs until the default set is stationary.

    Args:
        matrix: Weighted adjacency (exposures) of the slice.
        capitals: Equity capital of each bank.
        shocked_local_idx: Index of the initially shocked bank (assumed defaulted).
        loss_propagation: Fraction of defaulted exposure actually recovered
            (recovery rate); the complement hits counterparties as loss.

    Returns:
        Default probabilities in [0, 1] per node (1.0 for defaulted banks).
    """
    n = len(matrix)
    defaulted = [False] * n
    accumulated_loss = [0.0] * n

    # The shock instantaneously wipes out the shocked bank.
    defaulted[shocked_local_idx] = True

    for _ in range(n):  # cascade settles in at most n rounds
        changed = False
        for i in range(n):
            if defaulted[i] or accumulated_loss[i] >= capitals[i]:
                if not defaulted[i]:
                    defaulted[i] = True
                    changed = True
                continue
            # Loss absorbed from defaulted counterparties we lent to.
            loss = 0.0
            for j in range(n):
                if defaulted[j]:
                    exposure = matrix[i][j]
                    loss += exposure * (1.0 - loss_propagation)
            if loss > accumulated_loss[i]:
                accumulated_loss[i] = loss
                if loss >= capitals[i]:
                    defaulted[i] = True
                    changed = True
        if not changed:
            break

    # Soft score: 1.0 for defaults, ratio of loss-to-capital otherwise.
    scores: List[float] = []
    for i in range(n):
        if defaulted[i]:
            scores.append(1.0)
        else:
            scores.append(min(0.99, accumulated_loss[i] / max(capitals[i], 1e-9)))
    return scores


def page_rank_cascade(
    matrix: Sequence[Sequence[float]],
    shocked_local_idx: int,
    damping: float = 0.85,
) -> List[float]:
    """PageRank-style systemic-importance fallback over the exposure graph.

    Uses NetworkX PageRank on the directed exposure graph restricted to the
    slice and blends it with the shock proximity to produce a triage score.

    Returns:
        Normalized risk scores in [0, 1].
    """
    n = len(matrix)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0:
                graph.add_edge(i, j, weight=matrix[i][j])
    if len(graph) == 0:
        return [0.0] * n

    ranks = nx.pagerank(graph, weight="weight", alpha=damping, max_iter=200)
    max_rank = max(ranks.values()) or 1.0
    distances = nx.single_source_shortest_path_length(graph.to_undirected(), shocked_local_idx)

    scores: List[float] = []
    for i in range(n):
        proximity = 1.0 / (1.0 + distances.get(i, 4))
        blended = 0.5 * (ranks[i] / max_rank) + 0.5 * proximity
        scores.append(round(min(1.0, blended), 4))
    return scores


def graph_summary(graph: nx.DiGraph) -> Dict[str, float]:
    """Compute headline topology stats for the telemetry panel."""
    if graph.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0, "density": 0.0}
    return {
        "n_nodes": float(graph.number_of_nodes()),
        "n_edges": float(graph.number_of_edges()),
        "density": round(nx.density(graph), 4),
        "avg_clustering": round(nx.average_clustering(graph.to_undirected()), 4)
        if graph.number_of_nodes() > 2
        else 0.0,
    }
