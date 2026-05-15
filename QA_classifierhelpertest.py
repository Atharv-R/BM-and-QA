"""
QA-Compatible Discriminative Boltzmann Machine for MNIST Classification

Uses LABEL CHAINS: groups of physically-connected Zephyr qubits per class,
coupled with strong ferromagnetic interactions to encourage agreement.

KEY DIFFERENCE from classifierhelpertest.py:
  - NO node contraction / fusion — every node is a real physical qubit
  - Label chains must form connected subgraphs (physical couplers exist)
  - Chain couplings are explicit ferromagnetic terms in the energy function
  - Inference via majority vote on chain qubit states

QA deployment path:
  1. Train classically with chain couplings in the energy
  2. On QA hardware: fix pixel qubits (fix_variables), let label+hidden be free
  3. Sample via quantum annealing
  4. Decode label chains via majority vote

Architecture layout after relabeling:
  [0..num_pixels-1]                               = pixel visible nodes
  [num_pixels..num_pixels+total_label_nodes-1]     = label chain nodes (visible)
  [num_pixels+total_label_nodes..]                 = remaining hidden nodes
"""

import os
import time
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
import dwave_networkx as dnx

from bolmaqua import (
    graph_to_bm,
    train_boltzmann_machine_pcd,
    sample_from_bm,
    BoltzmannMachineGraph,
    CustomBoltzmannMachine,
    device,
    get_zephyr_positions,
    compute_pseudolikelihood,
    evaluate_reconstruction,
)


# =============================================================================
# UTILITY
# =============================================================================

def compute_chain_offsets(label_chains):
    """Cumulative offsets for variable-length label chains.

    Returns list of length len(label_chains) + 1, where offsets[c] is the
    start index of class c in the label encoding and offsets[-1] is the
    total number of label nodes.
    """
    offsets = [0]
    for chain in label_chains:
        offsets.append(offsets[-1] + len(chain))
    return offsets


# =============================================================================
# LABEL CHAIN SELECTION — Strategy Framework
# =============================================================================

# Registry: maps strategy name -> function
# Each strategy function has signature:
#   f(G, hidden_nodes_set, available, coverage_sets, target_nodes,
#     num_classes, nodes_per_label, rng, verbose) -> List[List[int]]
#
# To add a new strategy, use @register_chain_strategy('name')
CHAIN_SELECTION_STRATEGIES = {}


def register_chain_strategy(name):
    """Decorator to register a chain selection strategy."""
    def decorator(func):
        CHAIN_SELECTION_STRATEGIES[name] = func
        return func
    return decorator


# ---------------------------------------------------------------------------
# Strategy: connected_greedy
# ---------------------------------------------------------------------------
@register_chain_strategy('connected_greedy')
def _strategy_connected_greedy(
    G, hidden_nodes_set, available, coverage_sets, target_nodes,
    num_classes, nodes_per_label, rng, verbose, **kwargs,
):
    """
    Grow connected chains via round-robin greedy coverage maximization.

    Phase 1 — Seeding:
      For each class, pick the available hidden node with the highest
      target-neighborhood coverage as the seed.

    Phase 2 — Growth (round-robin):
      Repeat (nodes_per_label - 1) times:
        Sort classes by current chain coverage (ascending = worst first).
        For each class, find the frontier (available neighbors of the chain)
        and add the node with the highest marginal coverage gain.

    Properties:
      - Every chain is a connected subgraph (physical couplers between members)
      - Disjoint across classes (each node used by at most one class)
      - Fair allocation via round-robin with worst-first ordering
    """
    chains = [[] for _ in range(num_classes)]
    chain_sets = [set() for _ in range(num_classes)]
    chain_coverages = [set() for _ in range(num_classes)]

    # Phase 1: Seed each class with the best available node
    for c in range(num_classes):
        if not available:
            raise RuntimeError(
                f"Ran out of hidden nodes while seeding class {c}."
            )
        seed = max(available, key=lambda n: (
            len(coverage_sets[n]),
            G.degree(n),
            -n,
        ))
        chains[c].append(seed)
        chain_sets[c].add(seed)
        chain_coverages[c] = coverage_sets[seed].copy()
        available.remove(seed)

    # Phase 2: Round-robin growth
    for round_idx in range(nodes_per_label - 1):
        # Process worst-coverage classes first
        class_order = sorted(
            range(num_classes),
            key=lambda c: (len(chain_coverages[c]), c),
        )
        for c in class_order:
            if len(chains[c]) >= nodes_per_label:
                continue

            # Frontier: available hidden nodes adjacent to current chain
            frontier = set()
            for node in chains[c]:
                for nbr in G.neighbors(node):
                    if nbr in available and nbr in hidden_nodes_set:
                        frontier.add(nbr)

            if not frontier:
                if verbose:
                    print(
                        f"  Warning: Chain for class {c} stuck at "
                        f"{len(chains[c])}/{nodes_per_label} nodes "
                        f"(no connected available nodes in frontier)"
                    )
                continue

            # Pick frontier node with best marginal coverage
            best = max(frontier, key=lambda n: (
                len(coverage_sets[n] - chain_coverages[c]),
                len(coverage_sets[n]),
                G.degree(n),
                -n,
            ))

            chains[c].append(best)
            chain_sets[c].add(best)
            chain_coverages[c] |= coverage_sets[best]
            available.remove(best)

    return chains


# ---------------------------------------------------------------------------
# Strategy: random_walk
# ---------------------------------------------------------------------------
@register_chain_strategy('random_walk')
def _strategy_random_walk(
    G, hidden_nodes_set, available, coverage_sets, target_nodes,
    num_classes, nodes_per_label, rng, verbose, **kwargs,
):
    """
    Grow chains via random walk from high-coverage seeds.

    Useful as a baseline: fast, stochastic, but no coverage optimization.
    Connectivity is guaranteed since each step extends to a neighbor.
    """
    chains = [[] for _ in range(num_classes)]
    chain_sets = [set() for _ in range(num_classes)]

    # Seed: same as connected_greedy (best coverage)
    for c in range(num_classes):
        if not available:
            raise RuntimeError(
                f"Ran out of hidden nodes while seeding class {c}."
            )
        seed = max(available, key=lambda n: (
            len(coverage_sets[n]),
            -n,
        ))
        chains[c].append(seed)
        chain_sets[c].add(seed)
        available.remove(seed)

    # Growth: random walk
    for round_idx in range(nodes_per_label - 1):
        for c in range(num_classes):
            if len(chains[c]) >= nodes_per_label:
                continue

            frontier = [
                nbr for node in chains[c] for nbr in G.neighbors(node)
                if nbr in available and nbr in hidden_nodes_set
            ]
            if not frontier:
                if verbose:
                    print(
                        f"  Warning: Chain for class {c} stuck at "
                        f"{len(chains[c])}/{nodes_per_label} nodes"
                    )
                continue

            chosen = frontier[rng.integers(len(frontier))]
            chains[c].append(chosen)
            chain_sets[c].add(chosen)
            available.remove(chosen)

    return chains


# ---------------------------------------------------------------------------
# Strategy: degree_weighted
# ---------------------------------------------------------------------------
@register_chain_strategy('degree_weighted')
def _strategy_degree_weighted(
    G, hidden_nodes_set, available, coverage_sets, target_nodes,
    num_classes, nodes_per_label, rng, verbose, **kwargs,
):
    """
    Like connected_greedy but prioritizes high-degree nodes.

    High degree in the Zephyr graph means more physical couplers, which
    translates to stronger information flow on QA hardware. This strategy
    balances coverage with QA-readiness.
    """
    chains = [[] for _ in range(num_classes)]
    chain_sets = [set() for _ in range(num_classes)]
    chain_coverages = [set() for _ in range(num_classes)]

    # Seed by degree first, coverage second
    for c in range(num_classes):
        if not available:
            raise RuntimeError(
                f"Ran out of hidden nodes while seeding class {c}."
            )
        seed = max(available, key=lambda n: (
            G.degree(n),
            len(coverage_sets[n]),
            -n,
        ))
        chains[c].append(seed)
        chain_sets[c].add(seed)
        chain_coverages[c] = coverage_sets[seed].copy()
        available.remove(seed)

    # Growth: round-robin, degree-weighted selection
    for round_idx in range(nodes_per_label - 1):
        class_order = sorted(
            range(num_classes),
            key=lambda c: (len(chain_coverages[c]), c),
        )
        for c in class_order:
            if len(chains[c]) >= nodes_per_label:
                continue

            frontier = set()
            for node in chains[c]:
                for nbr in G.neighbors(node):
                    if nbr in available and nbr in hidden_nodes_set:
                        frontier.add(nbr)

            if not frontier:
                if verbose:
                    print(
                        f"  Warning: Chain for class {c} stuck at "
                        f"{len(chains[c])}/{nodes_per_label} nodes"
                    )
                continue

            # Weight degree more heavily
            best = max(frontier, key=lambda n: (
                G.degree(n),
                len(coverage_sets[n] - chain_coverages[c]),
                len(coverage_sets[n]),
                -n,
            ))

            chains[c].append(best)
            chain_sets[c].add(best)
            chain_coverages[c] |= coverage_sets[best]
            available.remove(best)

    return chains


# ---------------------------------------------------------------------------
# Strategy: seed_and_connect
# ---------------------------------------------------------------------------
@register_chain_strategy('seed_and_connect')
def _strategy_seed_and_connect(
    G, hidden_nodes_set, available, coverage_sets, target_nodes,
    num_classes, nodes_per_label, rng, verbose,
    num_seeds_per_label=3, **kwargs,
):
    """
    Seed for max 2-hop pixel coverage, then grow connected chains.

    Phase 1 — 2-hop pixel coverage:
      For each available hidden node, compute the set of pixel nodes
      reachable within 2 hops (direct neighbors + neighbors-of-neighbors).

    Phase 2 — Seed + connected growth (round-robin, worst-coverage-first):
      Seed each class with the best 2-hop coverage node, then grow via
      frontier expansion (like connected_greedy) using marginal 2-hop
      pixel coverage as the growth metric. Every added node is on the
      frontier of the current chain, guaranteeing connectivity by
      construction.

    This achieves the same goal as multi-seed placement (spread label
    nodes for image coverage) but avoids the connectivity problems of
    pre-placing spread seeds — the coverage metric naturally drives
    chains toward uncovered pixel regions.

    Chain length = nodes_per_label (same as other strategies).
    num_seeds_per_label is accepted but unused (kept for API compat).
    """
    min_hidden = min(hidden_nodes_set)
    pixel_set = set(range(min_hidden))

    if verbose:
        print(f"  Computing 2-hop pixel coverage for {len(available)} "
              f"hidden nodes...")

    # Phase 1: 2-hop pixel coverage per hidden node
    # Only count paths through hidden intermediaries: h → hidden → pixel
    # This ensures high-coverage nodes are well-connected in the hidden
    # subgraph (not just adjacent to many pixels directly).
    coverage_2hop = {}
    for h in available:
        neighbors = set(G.neighbors(h))
        # Direct pixel neighbors (1-hop)
        direct_pixels = neighbors & pixel_set
        # 2-hop via hidden neighbors only
        hidden_neighbors = neighbors & hidden_nodes_set
        two_hop_pixels = set(direct_pixels)
        for hh in hidden_neighbors:
            two_hop_pixels |= (set(G.neighbors(hh)) & pixel_set)
        coverage_2hop[h] = two_hop_pixels

    # Restrict to the giant connected component of the hidden subgraph
    # so seeds are never placed on isolated nodes that can't grow.
    H_hidden = G.subgraph(available)
    giant_component = max(nx.connected_components(H_hidden), key=len)
    excluded = available - giant_component
    available = available & giant_component

    if verbose and excluded:
        print(f"  Restricted to giant hidden component: "
              f"{len(available)} nodes "
              f"(excluded {len(excluded)} isolated/small-component nodes)")

    # Phase 2: Seed + connected growth
    chains = [[] for _ in range(num_classes)]
    chain_sets = [set() for _ in range(num_classes)]
    chain_pixel_coverage = [set() for _ in range(num_classes)]

    if verbose:
        print(f"  Growing coverage-optimized connected chains "
              f"(target: {nodes_per_label} nodes/class)...")

    # Seed each class with the best 2-hop coverage node
    for c in range(num_classes):
        if not available:
            raise RuntimeError(
                f"Ran out of hidden nodes while seeding class {c}."
            )
        seed = max(available, key=lambda n: (
            len(coverage_2hop.get(n, set())),
            G.degree(n),
            -n,
        ))
        chains[c].append(seed)
        chain_sets[c].add(seed)
        chain_pixel_coverage[c] = coverage_2hop.get(seed, set()).copy()
        available.remove(seed)

    # Growth: round-robin, worst-coverage-first
    stuck_classes = set()
    for round_idx in range(nodes_per_label - 1):
        class_order = sorted(
            range(num_classes),
            key=lambda c: (len(chain_pixel_coverage[c]), c),
        )
        for c in class_order:
            if len(chains[c]) >= nodes_per_label:
                continue

            frontier = set()
            for node in chains[c]:
                for nbr in G.neighbors(node):
                    if nbr in available and nbr in hidden_nodes_set:
                        frontier.add(nbr)

            if not frontier:
                if verbose and c not in stuck_classes:
                    print(
                        f"  Warning: Chain for class {c} stuck at "
                        f"{len(chains[c])}/{nodes_per_label} nodes "
                        f"(no connected available nodes in frontier)"
                    )
                    stuck_classes.add(c)
                continue

            # Pick frontier node with best marginal 2-hop pixel coverage
            best = max(frontier, key=lambda n: (
                len(coverage_2hop.get(n, set()) - chain_pixel_coverage[c]),
                len(coverage_2hop.get(n, set())),
                G.degree(n),
                -n,
            ))

            chains[c].append(best)
            chain_sets[c].add(best)
            chain_pixel_coverage[c] |= coverage_2hop.get(best, set())
            available.remove(best)

    if verbose:
        for c in range(num_classes):
            print(f"    Class {c}: {len(chains[c])} nodes, "
                  f"2-hop pixel coverage: "
                  f"{len(chain_pixel_coverage[c])}/{len(pixel_set)}")

    return chains


@register_chain_strategy('ilp_coverage')
def _strategy_ilp_coverage(
    G, hidden_nodes_set, available, coverage_sets, target_nodes,
    num_classes, nodes_per_label, rng, verbose,
    gurobi_time_limit=300, **kwargs,
):
    """
    ILP-based chain placement maximizing minimum 2-hop pixel coverage.

    Solves a MILP with Gurobi:
        max  t
        s.t. each hidden node used by at most one class  (disjointness)
             each class gets exactly nodes_per_label nodes (budget)
             pixel p covered by class c iff ≥1 chain node 2-hop-reaches p
             t ≤ (covered pixels for each class)              (max-min)
             each class's chain is connected   (flow constraints)

    Falls back to connected_greedy if Gurobi is unavailable or no connected
    solution is found within the time limit.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        if verbose:
            print("  gurobipy not available — falling back to connected_greedy")
        return _strategy_connected_greedy(
            G, hidden_nodes_set, available, coverage_sets, target_nodes,
            num_classes, nodes_per_label, rng, verbose,
        )

    import hashlib
    import json
    import time as _time

    min_hidden = min(hidden_nodes_set)
    pixel_set = set(range(min_hidden))
    pixel_list = sorted(pixel_set)
    avail_list = sorted(available)
    avail_set = set(avail_list)

    if verbose:
        print(f"  Computing 2-hop pixel coverage (via hidden) for "
              f"{len(avail_list)} nodes...")

    # 2-hop coverage: h → hidden_neighbor → pixel (+ direct h → pixel)
    cover_2hop = {}
    for h in avail_list:
        nbrs = set(G.neighbors(h))
        pixels = nbrs & pixel_set
        for hh in nbrs & hidden_nodes_set:
            pixels = pixels | (set(G.neighbors(hh)) & pixel_set)
        cover_2hop[h] = pixels

    if verbose:
        cov_sizes = [len(cover_2hop[h]) for h in avail_list]
        print(f"  Coverage stats: min={min(cov_sizes)}, "
              f"mean={np.mean(cov_sizes):.0f}, max={max(cov_sizes)}")

    # --- Warm start from connected_greedy ---
    if verbose:
        print(f"  Computing warm start (connected_greedy)...")
    warm_avail = set(avail_list)
    warm_chains = _strategy_connected_greedy(
        G, hidden_nodes_set, warm_avail, coverage_sets, target_nodes,
        num_classes, nodes_per_label, rng, verbose=False,
    )
    warm_chain_sets = [set(c) for c in warm_chains]
    warm_coverages = []
    for c in range(num_classes):
        cov = set()
        for h in warm_chains[c]:
            cov |= cover_2hop.get(h, set())
        warm_coverages.append(len(cov))
    warm_min_cov = min(warm_coverages)
    if verbose:
        print(f"  Warm start min coverage: {warm_min_cov}/{len(pixel_list)}")
        print(f"  Warm start per-class: {warm_coverages}")

    # --- Variable reduction ---
    # Full-size ILP (40K+ binaries) is intractable; reduce to top
    # candidates by coverage + their 1-hop hidden neighbours (connectors)
    # + warm-start nodes.  Keeps the ILP ≤ ~2-3K candidates.
    max_candidates = int(kwargs.get('max_candidates', 800))
    full_avail_count = len(avail_list)
    if full_avail_count > max_candidates + 200:
        sorted_by_cov = sorted(
            avail_list,
            key=lambda h: len(cover_2hop[h]),
            reverse=True,
        )
        candidate_set = set(sorted_by_cov[:max_candidates])
        # Add 1-hop hidden neighbours for potential connectivity
        for h in sorted_by_cov[:max_candidates]:
            for nbr in G.neighbors(h):
                if nbr in avail_set and nbr not in candidate_set:
                    candidate_set.add(nbr)
        # Ensure warm-start nodes are included
        for chain in warm_chains:
            candidate_set |= set(chain)
        if verbose:
            print(f"  Variable reduction: {full_avail_count} → "
                  f"{len(candidate_set)} candidates "
                  f"(top {max_candidates} + neighbours + warm start)")
        avail_list = sorted(candidate_set)
        avail_set = candidate_set

    # Reverse index (pixel → candidate hidden nodes) on reduced set
    pix_covered_by = {p: [] for p in pixel_list}
    for h in avail_list:
        for p in cover_2hop[h]:
            pix_covered_by[p].append(h)

    # --- Build MILP ---
    if verbose:
        print(f"  Building MILP: {len(avail_list)} candidates × "
              f"{num_classes} classes, {len(pixel_list)} pixels...")

    checkpoint_enabled = bool(kwargs.get('gurobi_checkpoint_enabled', True))
    checkpoint_dir = kwargs.get('gurobi_checkpoint_dir', 'gurobi_checkpoints')
    checkpoint_tag = str(kwargs.get('gurobi_checkpoint_tag', '') or '').strip()

    checkpoint_mst_path = None
    checkpoint_sol_path = None
    checkpoint_json_path = None
    checkpoint_meta = None

    if checkpoint_enabled:
        os.makedirs(checkpoint_dir, exist_ok=True)

        def _sha_items(items):
            hsh = hashlib.sha256()
            for item in items:
                hsh.update(str(item).encode('utf-8'))
                hsh.update(b'\n')
            return hsh.hexdigest()

        graph_edge_hash = _sha_items(
            (min(u, v), max(u, v)) for u, v in sorted(G.edges())
        )
        candidate_hash = _sha_items(avail_list)
        warm_chain_hash = _sha_items(
            (c, h) for c, chain in enumerate(warm_chains) for h in chain
        )
        checkpoint_meta = {
            'strategy': 'ilp_coverage',
            'solver': 'gurobi',
            'formulation': 'flow_connectivity_v1',
            'num_graph_nodes': G.number_of_nodes(),
            'num_graph_edges': G.number_of_edges(),
            'graph_edge_hash': graph_edge_hash,
            'num_visible_pixels': len(pixel_list),
            'num_hidden_nodes': len(hidden_nodes_set),
            'num_target_nodes': len(target_nodes),
            'num_classes': num_classes,
            'nodes_per_label': nodes_per_label,
            'max_candidates': max_candidates,
            'num_candidates': len(avail_list),
            'candidate_hash': candidate_hash,
            'roots': [],
            'warm_chain_hash': warm_chain_hash,
            'gurobi_seed': kwargs.get('gurobi_seed'),
        }
        checkpoint_fingerprint = hashlib.sha256(
            json.dumps(checkpoint_meta, sort_keys=True).encode('utf-8')
        ).hexdigest()[:16]
        clean_tag = ''.join(
            ch if ch.isalnum() or ch in ('-', '_') else '_'
            for ch in checkpoint_tag
        ).strip('_')
        name_parts = [
            'ilp_coverage',
            f'C{num_classes}',
            f'B{nodes_per_label}',
            f'V{len(pixel_list)}',
            f'H{len(hidden_nodes_set)}',
            f'N{len(avail_list)}',
        ]
        if clean_tag:
            name_parts.insert(1, clean_tag)
        checkpoint_base = '_'.join(name_parts) + f'_{checkpoint_fingerprint}'
        checkpoint_mst_path = os.path.join(checkpoint_dir, checkpoint_base + '.mst')
        checkpoint_sol_path = os.path.join(checkpoint_dir, checkpoint_base + '.sol')
        checkpoint_json_path = os.path.join(checkpoint_dir, checkpoint_base + '.json')
        if verbose:
            print(f"  Gurobi checkpoint base: {checkpoint_base}")

    # License convenience: prefer explicit GRB_LICENSE_FILE if already set,
    # otherwise use common local paths.
    if 'GRB_LICENSE_FILE' not in os.environ:
        license_candidates = [
            os.path.join(os.getcwd(), 'gurobi.lic'),
            os.path.expanduser('~/gurobi.lic'),
            os.path.expanduser('~/gurobi/gurobi.lic'),
        ]
        for lic_path in license_candidates:
            if os.path.exists(lic_path):
                os.environ['GRB_LICENSE_FILE'] = lic_path
                if verbose:
                    print(f"  Using Gurobi license: {lic_path}")
                break

    t_build_start = _time.time()
    m = gp.Model("chain_coverage")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = float(gurobi_time_limit)
    m.Params.MIPGap = 0.02

    # Optional tuning knobs
    if 'gurobi_threads' in kwargs:
        m.Params.Threads = int(kwargs['gurobi_threads'])
    if 'gurobi_seed' in kwargs:
        m.Params.Seed = int(kwargs['gurobi_seed'])

    # x[c,h] ∈ {0,1}: node h assigned to class c
    x = m.addVars(
        range(num_classes),
        avail_list,
        vtype=GRB.BINARY,
        name="x",
    )

    # y[c,p] ∈ {0,1}: pixel p 2-hop-covered by class c
    y = m.addVars(
        range(num_classes),
        pixel_list,
        vtype=GRB.BINARY,
        name="y",
    )

    # t: minimum pixel coverage across all classes
    t_var = m.addVar(
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=float(len(pixel_list)),
        name="t",
    )
    m.setObjective(t_var, GRB.MAXIMIZE)

    # C1: Disjointness — each hidden node in at most one chain
    for h in avail_list:
        m.addConstr(gp.quicksum(x[c, h] for c in range(num_classes)) <= 1)

    # C2: Budget — each class gets exactly nodes_per_label nodes
    for c in range(num_classes):
        m.addConstr(gp.quicksum(x[c, h] for h in avail_list) == nodes_per_label)

    # C3: Coverage linking — y[c,p] ≤ Σ x[c,h] over h that 2-hop-cover p
    for c in range(num_classes):
        for p in pixel_list:
            covering = pix_covered_by[p]
            if covering:
                m.addConstr(
                    y[c, p] <= gp.quicksum(x[c, h] for h in covering)
                )
            else:
                m.addConstr(y[c, p] == 0)

    # C4: Max-min — t ≤ (coverage of each class)
    for c in range(num_classes):
        m.addConstr(t_var <= gp.quicksum(y[c, p] for p in pixel_list))

    # C5: Flow-based connectivity —
    # Single-commodity flow from a fixed root per class ensures the
    # selected subgraph is connected.  Root = first warm-start node
    # (forced selected).  Each selected non-root node absorbs 1 unit;
    # the root emits B-1 units.  Flow travels only through selected
    # nodes (capacity constraint).
    B = nodes_per_label
    roots = {}
    for c in range(num_classes):
        rc = warm_chains[c][0]
        roots[c] = rc
        m.addConstr(x[c, rc] == 1, name=f"root_{c}")

    if checkpoint_meta is not None:
        checkpoint_meta['roots'] = [int(roots[c]) for c in range(num_classes)]

    # Adjacency within candidate set (for directed flow edges)
    nbr_in_cand = {}
    for h in avail_list:
        nbr_in_cand[h] = [j for j in G.neighbors(h) if j in avail_set]

    # Flow variables: f[c,i,j] ≥ 0 for each directed candidate edge
    flow_keys = [
        (c, h, j)
        for c in range(num_classes)
        for h in avail_list
        for j in nbr_in_cand[h]
    ]
    f = m.addVars(
        flow_keys,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=float(B - 1),
        name="f",
    )

    # Flow conservation
    for c in range(num_classes):
        rc = roots[c]
        for h in avail_list:
            inflow = gp.quicksum(f[c, j, h] for j in nbr_in_cand[h])
            outflow = gp.quicksum(f[c, h, j] for j in nbr_in_cand[h])
            if h == rc:
                m.addConstr(
                    outflow - inflow == B - 1,
                    name=f"flow_root_{c}",
                )
            else:
                m.addConstr(
                    inflow - outflow == x[c, h],
                    name=f"flow_{c}_{h}",
                )

    # Flow capacity: only selected nodes can forward flow
    for c, h, j in flow_keys:
        m.addConstr(
            f[c, h, j] <= (B - 1) * x[c, h],
            name=f"cap_{c}_{h}_{j}",
        )

    # --- Warm start solution (including flow values) ---
    for c in range(num_classes):
        # x values
        for h in avail_list:
            x[c, h].Start = 1.0 if h in warm_chain_sets[c] else 0.0
        # y values
        warm_pix_cov_c = set()
        for h in warm_chains[c]:
            warm_pix_cov_c |= cover_2hop.get(h, set())
        for p in pixel_list:
            y[c, p].Start = 1.0 if p in warm_pix_cov_c else 0.0
        # Flow values from BFS tree rooted at root
        rc = roots[c]
        chain_subg = G.subgraph(warm_chains[c])
        tree = nx.bfs_tree(chain_subg, rc)
        subtree_sz = {}
        for nd in reversed(list(nx.topological_sort(tree))):
            subtree_sz[nd] = 1 + sum(
                subtree_sz.get(ch, 0) for ch in tree.successors(nd)
            )
        for parent in tree.nodes():
            for child in tree.successors(parent):
                if (c, parent, child) in f:
                    f[c, parent, child].Start = float(subtree_sz[child])
    t_var.Start = float(warm_min_cov)

    m.update()

    if checkpoint_enabled:
        loaded_checkpoint = None
        for candidate_path in (checkpoint_mst_path, checkpoint_sol_path):
            if candidate_path and os.path.exists(candidate_path):
                try:
                    m.read(candidate_path)
                    loaded_checkpoint = candidate_path
                    break
                except gp.GurobiError as exc:
                    if verbose:
                        print(f"  Could not load checkpoint {candidate_path}: {exc}")
        if verbose:
            if loaded_checkpoint:
                print(f"  Loaded Gurobi checkpoint: {loaded_checkpoint}")
            else:
                print("  No matching Gurobi checkpoint found; using greedy warm start")

    t_build = _time.time() - t_build_start
    if verbose:
        n_x = num_classes * len(avail_list)
        n_f = len(flow_keys)
        print(f"  Model built in {t_build:.1f}s  "
              f"({n_x} binary + {num_classes * len(pixel_list) + 1} cont "
              f"+ {n_f} flow vars)")

    # --- Solve ---
    if verbose:
        print(f"\n  Solving MILP (time limit {gurobi_time_limit}s)...")

    m.optimize()
    status = m.Status
    best_chains = None

    if m.SolCount > 0:
        chains = [[] for _ in range(num_classes)]
        for c in range(num_classes):
            for h in avail_list:
                if x[c, h].X > 0.5:
                    chains[c].append(h)

        obj_val = float(m.ObjVal)
        dual_bound = float(m.ObjBound)
        gap = float(m.MIPGap) if m.IsMIP else 0.0
        status_name = {
            GRB.OPTIMAL: 'OPTIMAL',
            GRB.TIME_LIMIT: 'TIME_LIMIT',
            GRB.INTERRUPTED: 'INTERRUPTED',
            GRB.SUBOPTIMAL: 'SUBOPTIMAL',
        }.get(status, str(status))

        coverages = []
        for c in range(num_classes):
            cov = set()
            for h in chains[c]:
                cov |= cover_2hop.get(h, set())
            coverages.append(len(cov))

        if verbose:
            print(f"  Status: {status_name}, gap: {gap:.2%}")
            print(f"  Best obj: {obj_val:.1f}, dual bound: {dual_bound:.1f}, "
                  f"actual min cov: {min(coverages)}")
            print(f"  Per-class 2-hop pixel coverage: {coverages}")
            print(f"  Chain lengths: {[len(c) for c in chains]}")

        if checkpoint_enabled:
            try:
                sol_saved = False
                try:
                    m.write(checkpoint_sol_path)
                    sol_saved = True
                except gp.GurobiError as exc:
                    if verbose:
                        print(f"  Could not save optional .sol checkpoint: {exc}")

                for var in m.getVars():
                    var.Start = var.X
                m.update()
                m.write(checkpoint_mst_path)
                checkpoint_payload = dict(checkpoint_meta or {})
                checkpoint_payload.update({
                    'saved_at_unix': _time.time(),
                    'gurobi_time_limit': gurobi_time_limit,
                    'status': status_name,
                    'objective': obj_val,
                    'dual_bound': dual_bound,
                    'mip_gap': gap,
                    'actual_min_2hop_pixel_coverage': min(coverages),
                    'per_class_2hop_pixel_coverage': coverages,
                    'chain_lengths': [len(c) for c in chains],
                    'chains_original_ids': [list(c) for c in chains],
                    'mst_path': checkpoint_mst_path,
                    'sol_path': checkpoint_sol_path if sol_saved else None,
                })
                with open(checkpoint_json_path, 'w') as f_json:
                    json.dump(checkpoint_payload, f_json, indent=2, sort_keys=True)
                if verbose:
                    print(f"  Saved Gurobi checkpoint: {checkpoint_mst_path}")
            except (OSError, gp.GurobiError) as exc:
                if verbose:
                    print(f"  Could not save Gurobi checkpoint: {exc}")

        # Verify connectivity (guaranteed by flow, but double-check)
        all_connected = True
        for c in range(num_classes):
            if len(chains[c]) > 1:
                if not nx.is_connected(G.subgraph(chains[c])):
                    all_connected = False
                    if verbose:
                        nc = nx.number_connected_components(
                            G.subgraph(chains[c]))
                        print(f"  WARNING: Class {c} has {nc} components!")

        if all_connected:
            best_chains = chains
            if verbose:
                print(f"  ✓ All chains connected!")
        elif verbose:
            print(f"  Connectivity violated — using warm start")
    elif verbose:
        status_name = {
            GRB.INFEASIBLE: 'INFEASIBLE',
            GRB.INF_OR_UNBD: 'INF_OR_UNBD',
            GRB.UNBOUNDED: 'UNBOUNDED',
        }.get(status, str(status))
        print(f"  No feasible solution (status={status_name}) — using warm start")

    if best_chains is None:
        if verbose:
            print(f"  Falling back to warm start (connected_greedy)")
        best_chains = warm_chains

    # Update available set (mirror what other strategies do)
    for chain in best_chains:
        available -= set(chain)

    return best_chains


# =============================================================================
# MAIN CHAIN SELECTION ENTRY POINT
# =============================================================================

def select_label_chains(
    G,
    visible_nodes,
    hidden_nodes,
    num_classes=10,
    nodes_per_label=3,
    strategy='connected_greedy',
    coverage_mode='hidden_only',
    seed=None,
    verbose=True,
    strategy_kwargs=None,
):
    """
    Select label chains from the hidden-node pool.

    Each class gets `nodes_per_label` physically-connected hidden qubits.
    The graph is NOT modified — no node contraction or fusion.

    Args:
        G: The relabeled graph (pixels=0..p-1, hidden=p..n-1)
        visible_nodes: List of pixel-visible node IDs
        hidden_nodes: List of hidden node IDs
        num_classes: Number of classes
        nodes_per_label: Chain length per class (number of physical qubits)
        strategy: Chain selection strategy name (see CHAIN_SELECTION_STRATEGIES)
        coverage_mode: 'hidden_only' or 'hidden_visible'
            - 'hidden_only': maximize coverage of other hidden nodes
            - 'hidden_visible': maximize coverage of hidden + pixel nodes
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        label_chains: List[List[int]] — one chain per class, each a list of
                      node IDs (in the input graph's labeling)
        chain_stats: Dict with selection statistics
    """
    if strategy == 'seed_and_connect':
        _n_seeds = (strategy_kwargs or {}).get('num_seeds_per_label', 3)
        min_label_nodes = num_classes * _n_seeds
    else:
        min_label_nodes = num_classes * nodes_per_label
    if min_label_nodes > len(hidden_nodes):
        raise ValueError(
            f"Need at least {min_label_nodes} hidden nodes for label chains, "
            f"but only {len(hidden_nodes)} available."
        )

    if strategy not in CHAIN_SELECTION_STRATEGIES:
        available_strategies = list(CHAIN_SELECTION_STRATEGIES.keys())
        raise ValueError(
            f"Unknown chain strategy '{strategy}'. "
            f"Available: {available_strategies}"
        )

    # Compute coverage targets
    hidden_set = set(hidden_nodes)
    visible_set = set(visible_nodes)
    if coverage_mode == 'hidden_only':
        target_nodes = hidden_set
    elif coverage_mode == 'hidden_visible':
        target_nodes = hidden_set | visible_set
    else:
        raise ValueError(
            f"Unknown coverage_mode='{coverage_mode}'. "
            f"Use 'hidden_only' or 'hidden_visible'."
        )

    # Precompute coverage sets: for each hidden node, which target nodes
    # are its direct neighbors?
    coverage_sets = {}
    for h in hidden_nodes:
        coverage_sets[h] = set(G.neighbors(h)) & target_nodes

    if verbose:
        print(f"\n{'='*70}")
        print(f"  LABEL CHAIN SELECTION")
        print(f"{'='*70}")
        print(f"  Strategy: {strategy}")
        print(f"  Coverage mode: {coverage_mode}")
        if strategy == 'seed_and_connect':
            print(f"  Classes: {num_classes}, Chain length: {nodes_per_label} "
                  f"(2-hop pixel coverage guided)")
        elif strategy == 'ilp_coverage':
            _tl = (strategy_kwargs or {}).get('gurobi_time_limit', 300)
            print(f"  Classes: {num_classes}, Chain length: {nodes_per_label} "
                f"(Gurobi max-min coverage, time limit: {_tl}s)")
            print(f"  Total label qubits: {num_classes * nodes_per_label}")
        else:
            print(f"  Classes: {num_classes}, Chain length: {nodes_per_label}")
            print(f"  Total label qubits: {num_classes * nodes_per_label}")
        print(f"  Target nodes for coverage: {len(target_nodes)}")

    rng = np.random.default_rng(seed)
    available = set(hidden_nodes)

    t0 = time.time()
    strategy_fn = CHAIN_SELECTION_STRATEGIES[strategy]
    label_chains = strategy_fn(
        G, hidden_set, available, coverage_sets, target_nodes,
        num_classes, nodes_per_label, rng, verbose,
        **(strategy_kwargs or {}),
    )
    elapsed = time.time() - t0

    # Validate results
    _validate_chains(G, label_chains, hidden_set, num_classes, verbose)

    # Compute statistics
    per_class_coverage = []
    for chain in label_chains:
        covered = set()
        for node in chain:
            covered |= coverage_sets[node]
        per_class_coverage.append(len(covered))

    chain_stats = {
        'strategy': strategy,
        'coverage_mode': coverage_mode,
        'nodes_per_label': nodes_per_label,
        'chain_lengths': [len(c) for c in label_chains],
        'num_classes': num_classes,
        'per_class_coverage': per_class_coverage,
        'min_coverage': min(per_class_coverage),
        'max_coverage': max(per_class_coverage),
        'mean_coverage': float(np.mean(per_class_coverage)),
        'total_label_nodes': sum(len(c) for c in label_chains),
        'time_seconds': elapsed,
        'chains_original_ids': [list(c) for c in label_chains],
    }

    if verbose:
        print(f"\n  Chain selection complete ({elapsed:.2f}s):")
        print(f"    Min class coverage: {chain_stats['min_coverage']}")
        print(f"    Max class coverage: {chain_stats['max_coverage']}")
        print(f"    Mean class coverage: {chain_stats['mean_coverage']:.1f}")
        for c, chain in enumerate(label_chains):
            print(f"    Class {c}: {len(chain)} nodes, "
                  f"coverage={per_class_coverage[c]}")
        print(f"{'='*70}\n")

    return label_chains, chain_stats


def _validate_chains(G, label_chains, hidden_set, num_classes, verbose):
    """Verify chain validity: connectivity, disjointness, membership."""
    all_chain_nodes = set()
    for c, chain in enumerate(label_chains):
        chain_set = set(chain)

        if len(chain) == 0:
            raise RuntimeError(f"Class {c} chain is empty!")

        # Check membership: all nodes must be hidden
        non_hidden = chain_set - hidden_set
        if non_hidden:
            raise RuntimeError(
                f"Class {c} chain contains non-hidden nodes: {non_hidden}"
            )

        # Check disjointness
        overlap = chain_set & all_chain_nodes
        if overlap:
            raise RuntimeError(
                f"Class {c} chain overlaps with another chain at nodes: {overlap}"
            )
        all_chain_nodes |= chain_set

        # Check connectivity
        if len(chain) > 1:
            subgraph = G.subgraph(chain)
            if not nx.is_connected(subgraph):
                components = list(nx.connected_components(subgraph))
                raise RuntimeError(
                    f"Class {c} chain is NOT connected! "
                    f"Components: {components}. "
                    f"This means physical couplers don't exist between all "
                    f"chain members — cannot form a ferromagnetic chain on QA."
                )


# =============================================================================
# GRAPH RELABELING (no contraction)
# =============================================================================

def relabel_with_label_chains(G, pixel_visible_nodes, hidden_nodes, label_chains):
    """
    Relabel graph so that label chain nodes become visible, right after pixels.

    Layout:
      [0..num_pixels-1]                            = pixel visible nodes
      [num_pixels..num_pixels+total_label-1]        = label chain nodes (visible)
      [num_pixels+total_label..]                    = remaining hidden nodes

    The graph structure is UNCHANGED — no edges added or removed,
    no nodes contracted. Only node IDs are remapped.

    Args:
        G: Graph with current labeling
        pixel_visible_nodes: List of pixel node IDs (in current labeling)
        hidden_nodes: List of hidden node IDs (in current labeling)
        label_chains: List[List[int]] — chains per class (current labeling)

    Returns:
        G_relabeled: Graph with new node IDs
        label_chains_relabeled: List[List[int]] — chains with new IDs
        remaining_hidden: List[int] — remaining hidden node IDs (new labeling)
        node_labels: Dict[int, str] — maps node ID to 'visible' or 'hidden'
        mapping: Dict[int, int] — old ID -> new ID
    """
    num_pixels = len(pixel_visible_nodes)
    label_nodes_flat = [n for chain in label_chains for n in chain]
    label_node_set = set(label_nodes_flat)
    total_label_nodes = len(label_nodes_flat)

    # Validate
    if len(label_node_set) != total_label_nodes:
        raise ValueError("Label chains must be disjoint (duplicate nodes found).")
    if not label_node_set.issubset(set(hidden_nodes)):
        raise ValueError("All label chain nodes must come from the hidden pool.")

    # Build mapping
    mapping = {}

    # Pixels keep 0..num_pixels-1
    for new_idx, old_node in enumerate(pixel_visible_nodes):
        mapping[old_node] = new_idx

    # Label chain nodes: num_pixels..num_pixels+total_label-1
    # Ordered by class, then by position within chain
    label_offset = num_pixels
    label_chains_relabeled = []
    for chain in label_chains:
        new_chain = []
        for old_node in chain:
            new_id = label_offset
            mapping[old_node] = new_id
            new_chain.append(new_id)
            label_offset += 1
        label_chains_relabeled.append(new_chain)

    # Remaining hidden nodes
    remaining_hidden_old = [n for n in hidden_nodes if n not in label_node_set]
    hidden_offset = label_offset
    for offset, old_node in enumerate(remaining_hidden_old):
        mapping[old_node] = hidden_offset + offset

    G_relabeled = nx.relabel_nodes(G, mapping, copy=True)

    remaining_hidden = list(range(hidden_offset, hidden_offset + len(remaining_hidden_old)))

    # Node labels
    node_labels = {}
    for i in range(num_pixels):
        node_labels[i] = 'visible'
    for i in range(num_pixels, num_pixels + total_label_nodes):
        node_labels[i] = 'visible'  # Label chain nodes are visible
    for i in remaining_hidden:
        node_labels[i] = 'hidden'

    print(f"\nRelabeling complete (no contraction):")
    print(f"  Pixels:     0..{num_pixels - 1}")
    chain_lengths = [len(c) for c in label_chains]
    if len(set(chain_lengths)) == 1:
        chain_desc = f"{len(label_chains)} chains x {chain_lengths[0]} nodes"
    else:
        chain_desc = (f"{len(label_chains)} chains, "
                      f"lengths {min(chain_lengths)}-{max(chain_lengths)}")
    print(f"  Label nodes: {num_pixels}..{num_pixels + total_label_nodes - 1} "
          f"({chain_desc})")
    print(f"  Hidden:     {hidden_offset}..{hidden_offset + len(remaining_hidden_old) - 1}")
    print(f"  Total nodes: {G_relabeled.number_of_nodes()} "
          f"(unchanged from {G.number_of_nodes()})")

    return G_relabeled, label_chains_relabeled, remaining_hidden, node_labels, mapping


# =============================================================================
# CHAIN COUPLING MANAGEMENT
# =============================================================================

def get_chain_edge_indices(G, label_chains_relabeled):
    """
    Find all intra-chain edges as (i, j) index pairs in the visible weight matrix.

    These are the edges where ferromagnetic chain couplings will be applied.
    Only returns pairs where an actual graph edge exists (physical coupler).

    Args:
        G: Relabeled graph
        label_chains_relabeled: Chains with relabeled IDs

    Returns:
        chain_edges: List of (i, j) tuples — indices into model.W_vv_raw
                     (visible-space indices equal graph node IDs for visible nodes)
    """
    chain_edges = []
    for chain in label_chains_relabeled:
        for a_idx in range(len(chain)):
            for b_idx in range(a_idx + 1, len(chain)):
                node_a, node_b = chain[a_idx], chain[b_idx]
                if G.has_edge(node_a, node_b):
                    chain_edges.append((node_a, node_b))
    return chain_edges


def initialize_chain_couplings(model, chain_edges, chain_strength=2.0):
    """
    Set ferromagnetic couplings between chain members in the model.

    In our BM energy: E = -0.5 v^T W_vv v - ...
    A positive W_vv[i,j] makes configurations where both v_i=1 and v_j=1
    lower energy, encouraging agreement (ferromagnetic coupling).

    For the D-Wave BQM (BINARY vartype), the quadratic coefficient is
    -W_vv[i,j] (negated). So positive W_vv -> negative BQM coefficient
    -> ferromagnetic on hardware. This is correct.

    Args:
        model: CustomBoltzmannMachine
        chain_edges: List of (i, j) index pairs from get_chain_edge_indices
        chain_strength: Positive value for ferromagnetic coupling strength
    """
    with torch.no_grad():
        for i, j in chain_edges:
            model.W_vv_raw.data[i, j] = chain_strength
            model.W_vv_raw.data[j, i] = chain_strength

    print(f"  Initialized {len(chain_edges)} chain couplings "
          f"to strength={chain_strength}")


def clamp_chain_couplings(model, chain_edges, chain_strength):
    """
    Re-apply fixed chain coupling values after an optimizer step.

    Call this after optimizer.step() when chain_mode='fixed' to prevent
    the optimizer from modifying chain coupling weights.

    Args:
        model: CustomBoltzmannMachine
        chain_edges: List of (i, j) pairs
        chain_strength: Target coupling strength
    """
    with torch.no_grad():
        for i, j in chain_edges:
            model.W_vv_raw.data[i, j] = chain_strength
            model.W_vv_raw.data[j, i] = chain_strength


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_classification_batch(
    pixels, labels, label_chains, num_classes, nodes_per_label,
):
    """
    Extend pixel data with label chain encoding for training.

    For a sample with true class c:
      - All nodes in label_chains[c] are set to 1
      - All nodes in other chains are set to 0

    This is a multi-hot encoding: each class has `nodes_per_label` bits.

    Args:
        pixels: (batch_size, num_pixels) tensor
        labels: (batch_size,) tensor with class indices 0..num_classes-1
        label_chains: List[List[int]] — relabeled chain node IDs (used only
                      for counting; the actual ordering is by class then node)
        num_classes: Number of classes
        nodes_per_label: Nodes per chain

    Returns:
        extended_visible: (batch_size, num_pixels + total_label_nodes) tensor
    """
    batch_size = pixels.shape[0]
    chain_offsets = compute_chain_offsets(label_chains)
    total_label_nodes = chain_offsets[-1]

    label_encoding = torch.zeros(
        batch_size, total_label_nodes, device=pixels.device,
    )

    for i in range(batch_size):
        c = labels[i].item()
        start = chain_offsets[c]
        end = chain_offsets[c + 1]
        label_encoding[i, start:end] = 1.0

    extended_visible = torch.cat([pixels, label_encoding], dim=1)
    return extended_visible


# =============================================================================
# SCORING / INFERENCE
# =============================================================================

def compute_class_scores_free_energy(
    model, pixels, label_chains, num_classes, nodes_per_label,
):
    """
    Score each class by negative free energy with that class's chain clamped ON.

    For class c:
      - Label chain c nodes = 1, all other chains = 0
      - Compute F(v) = free_energy([pixels | label_encoding])
      - Score = -F(v)  (higher is better)

    NOTE: free_energy() uses an RBM-style approximation that ignores HH
    couplings. This is an approximation, but it's fast and differentiable.

    Args:
        model: CustomBoltzmannMachine
        pixels: (batch_size, num_pixels) tensor
        label_chains: List of chains (for structure; not directly used)
        num_classes: Number of classes
        nodes_per_label: Nodes per chain

    Returns:
        class_scores: (batch_size, num_classes) tensor
    """
    chain_offsets = compute_chain_offsets(label_chains)
    total_label_nodes = chain_offsets[-1]
    batch_size = pixels.shape[0]
    scores = []

    for c in range(num_classes):
        label_vec = torch.zeros(
            batch_size, total_label_nodes, device=pixels.device,
        )
        start = chain_offsets[c]
        label_vec[:, start:chain_offsets[c + 1]] = 1.0

        candidate = torch.cat([pixels, label_vec], dim=1)
        scores.append(-model.free_energy(candidate))

    return torch.stack(scores, dim=1)


def classify_images_fast(
    model, test_pixels, label_chains, num_classes, nodes_per_label,
    num_gibbs_steps=50,
    inference_method='free_energy',
):
    """
    Classify images using the trained discriminative BM.

    Inference methods:
      'free_energy': Score each class by clamped free energy (fast, differentiable).
                     QA-approximate: uses analytical trace over hidden units.
      'mean_field':  Clamp pixels, run mean-field updates on label+hidden,
                     decode label activations via per-chain mean activation.
                     Not QA-native but good classical approximation.
      'gibbs':       Clamp pixels, run Gibbs sampling on label+hidden,
                     decode via majority vote on chain qubit states.
                     Closest to actual QA behavior.

    Args:
        model: Trained CustomBoltzmannMachine
        test_pixels: (batch_size, num_pixels) tensor
        label_chains: List[List[int]] — relabeled chain IDs
        num_classes: Number of classes
        nodes_per_label: Nodes per chain
        num_gibbs_steps: Steps for mean_field or gibbs methods
        inference_method: 'free_energy', 'mean_field', or 'gibbs'

    Returns:
        pred_classes: (batch_size,) tensor of predicted class indices
        confidences: (batch_size,) tensor of confidence values
    """
    was_training = model.training
    model.eval()

    batch_size = test_pixels.shape[0]
    num_pixels = test_pixels.shape[1]
    chain_offsets = compute_chain_offsets(label_chains)
    total_label_nodes = chain_offsets[-1]

    with torch.inference_mode():
        if inference_method == 'free_energy':
            class_scores = compute_class_scores_free_energy(
                model, test_pixels, label_chains,
                num_classes, nodes_per_label,
            )
            probabilities = torch.softmax(class_scores, dim=1)
            pred_classes = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]

        elif inference_method == 'mean_field':
            # Initialize: clamp pixels, randomize label+hidden
            extended_v = torch.cat([
                test_pixels,
                torch.rand(batch_size, total_label_nodes, device=device),
            ], dim=1)
            h = torch.rand(batch_size, model.num_hidden, device=device)

            # Iterate mean-field with pixels clamped
            for step in range(num_gibbs_steps):
                _, h = model.mean_field_update(
                    extended_v, h, update_v=False, update_h=True,
                )
                v_recon, _ = model.mean_field_update(
                    extended_v, h, update_v=True, update_h=False,
                )
                # Only update label portion, keep pixels clamped
                extended_v = torch.cat([
                    test_pixels,
                    v_recon[:, num_pixels:],
                ], dim=1)

            # Decode: per-chain mean activation
            label_activations = extended_v[:, num_pixels:]
            class_scores = torch.zeros(batch_size, num_classes, device=device)
            for c in range(num_classes):
                start = chain_offsets[c]
                end = chain_offsets[c + 1]
                class_scores[:, c] = label_activations[:, start:end].mean(dim=1)

            pred_classes = torch.argmax(class_scores, dim=1)
            confidences = torch.max(class_scores, dim=1)[0]

        elif inference_method == 'gibbs':
            # Initialize: clamp pixels, random label+hidden
            extended_v = torch.cat([
                test_pixels,
                torch.bernoulli(
                    torch.full((batch_size, total_label_nodes), 0.5, device=device)
                ),
            ], dim=1)
            h = torch.bernoulli(
                torch.full((batch_size, model.num_hidden), 0.5, device=device)
            )

            # Run Gibbs sampling with pixels clamped
            for step in range(num_gibbs_steps):
                extended_v, h = model.gibbs_sample_step(
                    extended_v, h,
                    update_v=True, update_h=True, track_grad=False,
                )
                # Re-clamp pixels
                extended_v = torch.cat([
                    test_pixels,
                    extended_v[:, num_pixels:],
                ], dim=1)

            # Decode: majority vote per chain
            label_states = extended_v[:, num_pixels:]
            class_scores = torch.zeros(batch_size, num_classes, device=device)
            for c in range(num_classes):
                start = chain_offsets[c]
                end = chain_offsets[c + 1]
                class_scores[:, c] = label_states[:, start:end].mean(dim=1)

            pred_classes = torch.argmax(class_scores, dim=1)
            confidences = torch.max(class_scores, dim=1)[0]

        else:
            raise ValueError(f"Unknown inference method: {inference_method}")

    if was_training:
        model.train()

    return pred_classes, confidences


# =============================================================================
# TRAINING
# =============================================================================

def train_classifier_bm(
    model, data_loader, optimizer,
    num_epochs, k_steps,
    label_chains, num_classes, nodes_per_label,
    classification_loss_weight=1.0,
    chain_edges=None,
    chain_strength=2.0,
    chain_mode='fixed',
    train_eval_loader=None,
    val_loader=None,
    validation_interval=5,
    validation_inference_method='free_energy',
    validation_num_gibbs_steps=10,
):
    """
    Train discriminative BM with label chains.

    Loss = CD_loss + classification_loss_weight * cross_entropy(scores, labels)

    If chain_mode='fixed': chain coupling weights are clamped back to
    chain_strength after every optimizer step.
    If chain_mode='trainable': chain couplings are initialized to chain_strength
    but allowed to evolve via gradients.

    Args:
        model: CustomBoltzmannMachine
        data_loader: DataLoader yielding (pixels, labels) batches
        optimizer: PyTorch optimizer
        num_epochs: Number of training epochs
        k_steps: CD-k steps
        label_chains: List[List[int]] — relabeled chain node IDs
        num_classes: Number of classes
        nodes_per_label: Nodes per chain
        classification_loss_weight: Weight for classification cross-entropy loss
        chain_edges: List of (i, j) pairs for chain couplings (needed if fixed)
        chain_strength: Ferromagnetic coupling strength
        chain_mode: 'fixed' or 'trainable'
        train_eval_loader: Optional DataLoader for periodic train accuracy
        val_loader: Optional DataLoader for periodic validation accuracy
        validation_interval: Evaluate train/val accuracy every N epochs
        validation_inference_method: Inference method for periodic accuracy
        validation_num_gibbs_steps: Gibbs/mean-field steps for periodic eval

    Returns:
        training_history: Dict with loss histories
    """
    loss_history = []
    classification_loss_history = []
    total_loss_history = []
    pll_values = []
    train_recon_mse_history = []
    train_recon_acc_history = []
    train_recon_bce_history = []
    validation_epochs = []
    train_accuracy_history = []
    val_accuracy_history = []

    print(f"\nTraining Classifier BM (epochs={num_epochs}, k_steps={k_steps}, "
          f"num_classes={num_classes}, nodes_per_label={nodes_per_label}, "
          f"chain_mode={chain_mode})...")

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        for batch_idx, (pixels, labels) in enumerate(data_loader):
            pixels = pixels.to(device)
            labels = labels.to(device)

            # Extend visible with label chain encoding
            extended_visible = prepare_classification_batch(
                pixels, labels, label_chains, num_classes, nodes_per_label,
            )

            optimizer.zero_grad(set_to_none=True)

            # CD loss on the full visible vector (pixels + labels)
            cd_loss, _ = model(extended_visible, k_steps=k_steps)

            # Classification loss via free-energy scoring
            class_scores = compute_class_scores_free_energy(
                model, pixels, label_chains, num_classes, nodes_per_label,
            )
            cls_loss = F.cross_entropy(class_scores, labels)

            total_loss = cd_loss + classification_loss_weight * cls_loss
            total_loss.backward()
            optimizer.step()

            # Fix chain couplings if in fixed mode
            if chain_mode == 'fixed' and chain_edges is not None:
                clamp_chain_couplings(model, chain_edges, chain_strength)

            epoch_loss += cd_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

            del extended_visible, cd_loss, cls_loss, class_scores
            del total_loss, pixels, labels

        avg_loss = epoch_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        loss_history.append(avg_loss)
        classification_loss_history.append(avg_cls_loss)
        total_loss_history.append(avg_total_loss)

        # Compute PLL and reconstruction metrics
        with torch.no_grad():
            sample_batch = next(iter(data_loader))
            sample_pixels = sample_batch[0].to(device)
            sample_labels = sample_batch[1].to(device)
            sample_extended = prepare_classification_batch(
                sample_pixels, sample_labels,
                label_chains, num_classes, nodes_per_label,
            )
            pll = compute_pseudolikelihood(model, sample_extended, num_samples=50)
            pll_values.append(pll)

            train_metrics = evaluate_reconstruction(
                model, sample_extended, num_samples=50,
            )
            train_recon_mse_history.append(train_metrics['mse'])
            train_recon_acc_history.append(train_metrics['accuracy'])
            train_recon_bce_history.append(train_metrics['bce'])

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"CD: {avg_loss:.4f} | Cls: {avg_cls_loss:.4f} | "
            f"Total: {avg_total_loss:.4f} | PLL: {pll:.4f} | "
            f"Recon MSE: {train_recon_mse_history[-1]:.4f} | "
            f"Recon Acc: {train_recon_acc_history[-1]:.4f}"
        )

        should_validate = (
            val_loader is not None and
            validation_interval > 0 and
            ((epoch + 1) % validation_interval == 0 or epoch + 1 == num_epochs)
        )
        if should_validate:
            eval_train_loader = train_eval_loader if train_eval_loader is not None else data_loader
            train_acc, _ = evaluate_classifier(
                model,
                eval_train_loader,
                label_chains,
                num_classes=num_classes,
                nodes_per_label=nodes_per_label,
                num_gibbs_steps=validation_num_gibbs_steps,
                inference_method=validation_inference_method,
                verbose=False,
            )
            val_acc, _ = evaluate_classifier(
                model,
                val_loader,
                label_chains,
                num_classes=num_classes,
                nodes_per_label=nodes_per_label,
                num_gibbs_steps=validation_num_gibbs_steps,
                inference_method=validation_inference_method,
                verbose=False,
            )
            validation_epochs.append(epoch + 1)
            train_accuracy_history.append(train_acc)
            val_accuracy_history.append(val_acc)
            print(
                f"  Validation @ epoch {epoch+1}: "
                f"train_acc={100*train_acc:.2f}% | "
                f"val_acc={100*val_acc:.2f}%"
            )

    return {
        'pcd_loss': loss_history,
        'classification_loss': classification_loss_history,
        'total_loss': total_loss_history,
        'pll': pll_values,
        'validation_epochs': validation_epochs,
        'train_accuracy': train_accuracy_history,
        'val_accuracy': val_accuracy_history,
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_classifier(
    model, test_loader, label_chains,
    num_classes=10, nodes_per_label=3,
    num_gibbs_steps=100,
    inference_method='free_energy',
    verbose=True,
):
    """
    Evaluate classification accuracy on a full dataset.

    Args:
        model: Trained CustomBoltzmannMachine
        test_loader: DataLoader yielding (pixels, labels)
        label_chains: List[List[int]] — relabeled chain IDs
        num_classes: Number of classes
        nodes_per_label: Nodes per chain
        num_gibbs_steps: Steps for mean_field/gibbs inference
        inference_method: 'free_energy', 'mean_field', or 'gibbs'

    Returns:
        accuracy: Overall accuracy (float)
        per_class_acc: Dict[int, float] — per-class accuracy
    """
    was_training = model.training
    model.eval()
    all_preds = []
    all_labels = []

    if verbose:
        print(f"\nEvaluating classifier (method={inference_method}, "
              f"steps={num_gibbs_steps}, nodes_per_label={nodes_per_label})...")

    with torch.inference_mode():
        for batch_idx, (pixels, labels) in enumerate(test_loader):
            pixels = pixels.to(device)
            labels = labels.to(device)

            preds, confs = classify_images_fast(
                model, pixels, label_chains,
                num_classes, nodes_per_label,
                num_gibbs_steps=num_gibbs_steps,
                inference_method=inference_method,
            )

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            if verbose and (batch_idx + 1) % max(1, len(test_loader) // 10) == 0:
                print(f"  Processed {batch_idx+1}/{len(test_loader)} batches...")

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    correct = (all_preds == all_labels).sum().item()
    total = len(all_labels)
    accuracy = correct / total

    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            class_correct = (all_preds[mask] == all_labels[mask]).sum().item()
            per_class_acc[c] = class_correct / mask.sum().item()
        else:
            per_class_acc[c] = 0.0

    if verbose:
        print(f"\n  Overall Accuracy: {100*accuracy:.2f}%")
        for c in range(num_classes):
            print(f"  Class {c} Accuracy: {100*per_class_acc[c]:.2f}%")

    if was_training:
        model.train()

    return accuracy, per_class_acc


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def analyze_architecture(G, visible_nodes, hidden_nodes, name="Architecture"):
    """Print diagnostic statistics about the V/H partition quality."""
    visible_set = set(visible_nodes)
    vh_edges = vv_edges = hh_edges = 0
    vh_deg_v = {n: 0 for n in visible_nodes}
    vh_deg_h = {n: 0 for n in hidden_nodes}

    for u, v in G.edges():
        u_vis, v_vis = u in visible_set, v in visible_set
        if u_vis and v_vis:
            vv_edges += 1
        elif not u_vis and not v_vis:
            hh_edges += 1
        else:
            vh_edges += 1
            if u_vis:
                vh_deg_v[u] += 1
                vh_deg_h[v] += 1
            else:
                vh_deg_v[v] += 1
                vh_deg_h[u] += 1

    v_degs = list(vh_deg_v.values())
    h_degs = list(vh_deg_h.values())

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Visible: {len(visible_nodes)}  |  Hidden: {len(hidden_nodes)}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  VH edges: {vh_edges} ({100*vh_edges/max(1, G.number_of_edges()):.1f}%)")
    print(f"  VV edges: {vv_edges} ({100*vv_edges/max(1, G.number_of_edges()):.1f}%)")
    print(f"  HH edges: {hh_edges} ({100*hh_edges/max(1, G.number_of_edges()):.1f}%)")
    print(f"  VH degree per visible: min={min(v_degs)} mean={np.mean(v_degs):.2f} max={max(v_degs)}")
    print(f"  VH degree per hidden:  min={min(h_degs)} mean={np.mean(h_degs):.2f} max={max(h_degs)}")
    zero_v = sum(1 for d in v_degs if d == 0)
    zero_h = sum(1 for d in h_degs if d == 0)
    if zero_v:
        print(f"  Warning: {zero_v} visible nodes have 0 hidden neighbors!")
    if zero_h:
        print(f"  Warning: {zero_h} hidden nodes have 0 visible neighbors!")
    print(f"{'='*60}\n")
    return {
        'vh': vh_edges, 'vv': vv_edges, 'hh': hh_edges,
        'vh_deg_v': v_degs, 'vh_deg_h': h_degs,
    }


def analyze_label_chains(G, label_chains, num_pixels, num_classes, nodes_per_label=None):
    """
    Print detailed statistics about label chains for QA readiness.

    Reports per-chain: connectivity, intra-chain edges, degree to pixels,
    degree to hidden, total degree.
    """
    total_label_nodes = sum(len(c) for c in label_chains)
    hidden_start = num_pixels + total_label_nodes

    print(f"\n{'='*70}")
    print("  LABEL CHAIN ANALYSIS (QA-Compatible)")
    print(f"{'='*70}")
    chain_lengths = [len(c) for c in label_chains]
    if len(set(chain_lengths)) == 1:
        chain_desc = f"{num_classes} classes x {chain_lengths[0]} qubits/class"
    else:
        chain_desc = (f"{num_classes} classes, "
                      f"lengths {min(chain_lengths)}-{max(chain_lengths)}")
    print(f"  Total label qubits: {total_label_nodes} ({chain_desc})")

    for c, chain in enumerate(label_chains):
        # Intra-chain edges (physical couplers between chain members)
        chain_set = set(chain)
        intra_edges = sum(
            1 for a in chain for b in chain
            if a < b and G.has_edge(a, b)
        )
        max_tree_edges = len(chain) - 1

        # Connections to pixels, hidden, and other label chains
        pixel_conns = 0
        hidden_conns = 0
        other_label_conns = 0
        for node in chain:
            for nbr in G.neighbors(node):
                if nbr < num_pixels:
                    pixel_conns += 1
                elif nbr >= hidden_start:
                    hidden_conns += 1
                elif nbr not in chain_set:
                    other_label_conns += 1

        print(f"\n  Class {c} chain: nodes {chain}")
        print(f"    Intra-chain edges: {intra_edges} "
              f"(min for connectivity: {max_tree_edges})")
        print(f"    Pixel connections (total): {pixel_conns}")
        print(f"    Hidden connections (total): {hidden_conns}")
        print(f"    Cross-chain label connections: {other_label_conns}")

        for node in chain:
            deg = G.degree(node)
            print(f"      Node {node}: degree={deg}")

    print(f"{'='*70}\n")
