'''
ARCH4: Embedding-Based Hybrid Architecture

Strategy: Start from a minor-embedded complete bipartite graph (RBM skeleton)
on the Zephyr topology, then break chains and expand the visible set to cover
all image pixels. The initial "core" visible nodes form a sub-grid of the full
image, capturing global structure. Additional visible nodes are placed nearby
in the graph, acting as local detail nodes.

Rationale: This combines the strengths of RBM-like connectivity (dense VH
coupling for the core nodes) with the custom-BM approach (using all physical
qubits, no embedding overhead). The core visible nodes see many hidden
neighbours (inherited from the bipartite embedding), while expansion nodes
fill in the remaining pixels with sparser but more local connectivity.

Phase 1 — Core:
  Embed K_{g², g²} (g = initial_grid_size) into Z(K) using minorminer.
  Break chains: one qubit per visible chain → "anchor" visible node.
  Map anchors to a uniformly-spaced sub-grid of the target image.

Phase 2 — Expansion:
  Select (n_pixels − g²) additional visible nodes from remaining qubits.
  Two criteria (user-selectable):
    "graph_distance" — prefer qubits with smallest BFS distance to any anchor
    "connectivity"   — prefer qubits with most non-visible neighbours

Phase 3 — Pixel assignment:
  Anchor nodes keep their sub-grid pixel positions.
  Expansion nodes are matched to remaining pixels via spatial proximity
  in the Zephyr layout.
'''

#%% defs

import os
import sys
import json
import time
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.data import TensorDataset, DataLoader
import dwave_networkx as dnx
import minorminer

from bolmaqua import (
    graph_to_bm,
    train_boltzmann_machine_pcd,
    sample_from_bm,
    BM_SimAnn_Sampler,
    BM_Neal_Sampler,
    GRID_SHAPE,
    device,
    get_zephyr_positions,
    relabel_visible_first,
)

ARCH_NAME = "embedding_hybrid"
ARCH_LABEL = "ARCH4: Embedding-Based Hybrid"


def load_data(grid_shape=GRID_SHAPE):
    """Load downsized MNIST (digits 0 and 1), binarized."""
    try:
        train_feats = np.load(f'mnist{grid_shape[0]}x{grid_shape[1]}_trainfeats.npy')
        train_labels = np.load(f'mnist{grid_shape[0]}x{grid_shape[1]}_trainlabels.npy')
    except FileNotFoundError:
        print(f"Error: Could not find mnist{grid_shape[0]}x{grid_shape[1]} data files.")
        return None
    mask = (train_labels == 0) | (train_labels == 1)
    train_feats = train_feats[mask]
    train_feats = (train_feats > 0.5).astype(np.float32)
    return torch.from_numpy(train_feats)


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
                vh_deg_v[u] += 1; vh_deg_h[v] += 1
            else:
                vh_deg_v[v] += 1; vh_deg_h[u] += 1

    v_degs = list(vh_deg_v.values())
    h_degs = list(vh_deg_h.values())

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Visible: {len(visible_nodes)}  |  Hidden: {len(hidden_nodes)}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  VH edges: {vh_edges} ({100*vh_edges/max(1,G.number_of_edges()):.1f}%)")
    print(f"  VV edges: {vv_edges} ({100*vv_edges/max(1,G.number_of_edges()):.1f}%)")
    print(f"  HH edges: {hh_edges} ({100*hh_edges/max(1,G.number_of_edges()):.1f}%)")
    print(f"  VH degree per visible: min={min(v_degs)} mean={np.mean(v_degs):.2f} max={max(v_degs)}")
    print(f"  VH degree per hidden:  min={min(h_degs)} mean={np.mean(h_degs):.2f} max={max(h_degs)}")
    zero_v = sum(1 for d in v_degs if d == 0)
    zero_h = sum(1 for d in h_degs if d == 0)
    if zero_v:
        print(f"  ⚠️  {zero_v} visible nodes have 0 hidden neighbors!")
    if zero_h:
        print(f"  ⚠️  {zero_h} hidden nodes have 0 visible neighbors!")
    print(f"{'='*60}\n")
    return {'vh': vh_edges, 'vv': vv_edges, 'hh': hh_edges,
            'vh_deg_v': v_degs, 'vh_deg_h': h_degs}


# =========================================================================
# Sub-grid computation
# =========================================================================

def compute_initial_subgrid(grid_shape, initial_grid_size):
    """
    Compute the pixel positions of the initial sub-grid within the full image.

    Places initial_grid_size × initial_grid_size pixels approximately uniformly
    in the grid, with border ≈ spacing between nodes. Interior placement is
    preferred. Positions are returned in row-major order.

    Args:
        grid_shape: (nrows, ncols) of the full image (e.g., (28, 28))
        initial_grid_size: side length g of the initial square grid

    Returns:
        initial_pixels: list of (row, col) positions for the g² initial pixels
        remaining_pixels: list of (row, col) for all other pixels
    """
    nrows, ncols = grid_shape
    g = initial_grid_size

    # Compute approximately uniform spacing.
    # We want border ≈ inter-node spacing.
    # With g nodes and spacing s, border b:
    #   b + (g-1)*s + b = nrows - 1  (index range)
    #   If b ≈ s: (g+1)*s ≈ nrows - 1
    # Use linspace with half-spacing border on each side.
    def uniform_positions(n_pts, extent):
        """Place n_pts positions with border ≈ gap in [0, extent-1]."""
        # total_span = extent - 1 pixel indices
        # gap = total_span / (n_pts + 1)  → puts border=gap at each end
        # But: if that yields gap < 1, fall back to linspace without border
        if n_pts >= extent:
            return np.round(np.linspace(0, extent - 1, n_pts)).astype(int)
        gap = (extent - 1) / (n_pts + 1)
        positions = np.round(np.arange(1, n_pts + 1) * gap).astype(int)
        return positions

    row_pos = uniform_positions(g, nrows)
    col_pos = uniform_positions(g, ncols)

    initial_pixels = []
    for r in row_pos:
        for c in col_pos:
            initial_pixels.append((int(r), int(c)))

    all_pixels = set((r, c) for r in range(nrows) for c in range(ncols))
    initial_set = set(initial_pixels)
    remaining_pixels = sorted(all_pixels - initial_set)  # row-major order

    return initial_pixels, remaining_pixels


# =========================================================================
# Embedding & chain breaking
# =========================================================================

def find_bipartite_embedding(G_zephyr, n_logical_visible, n_logical_hidden=None,
                             timeout=60, tries=10):
    """
    Find a minor embedding of K_{n_vis, n_hid} into the Zephyr graph.

    Args:
        G_zephyr: target Zephyr graph
        n_logical_visible: number of visible logical nodes (g²)
        n_logical_hidden: number of hidden logical nodes (default = n_logical_visible)
        timeout: seconds per attempt for minorminer
        tries: number of random restart attempts

    Returns:
        embedding: dict mapping logical_node → list_of_physical_qubits (chain)
                   Logical nodes 0..n_vis-1 are "visible", n_vis..n_vis+n_hid-1 are "hidden".
                   Returns None if no embedding found.
    """
    if n_logical_hidden is None:
        n_logical_hidden = n_logical_visible

    B = nx.complete_bipartite_graph(n_logical_visible, n_logical_hidden)
    print(f"  Embedding K_{{{n_logical_visible},{n_logical_hidden}}} "
          f"({n_logical_visible + n_logical_hidden} logical nodes) "
          f"into Z with {G_zephyr.number_of_nodes()} physical qubits...")

    embedding = minorminer.find_embedding(B, G_zephyr, timeout=timeout, tries=tries)

    if not embedding:
        print(f"  ⚠️  No embedding found after {tries} tries ({timeout}s each)")
        return None

    chain_lens = [len(c) for c in embedding.values()]
    total_used = sum(chain_lens)
    print(f"  Embedding found: max chain={max(chain_lens)}, "
          f"avg chain={sum(chain_lens)/len(chain_lens):.1f}, "
          f"total qubits used={total_used}/{G_zephyr.number_of_nodes()}")

    return embedding


def break_chains_and_select_anchors(G_zephyr, embedding, n_logical_visible):
    """
    Break embedding chains and select one anchor qubit per visible logical node.

    For each visible chain, the qubit with the highest degree in the full
    Zephyr graph is chosen as the anchor (most connections → best integration
    into the native topology).

    Args:
        G_zephyr: full Zephyr graph
        embedding: dict from find_bipartite_embedding
        n_logical_visible: number of visible logical nodes

    Returns:
        anchor_nodes: list of n_logical_visible physical qubit IDs
                      (one per visible logical node, in order)
        chain_info: dict with chain statistics
    """
    anchor_nodes = []

    for logical_v in range(n_logical_visible):
        chain = embedding[logical_v]
        # Pick the qubit with highest Zephyr degree (breaks ties arbitrarily)
        best_qubit = max(chain, key=lambda q: G_zephyr.degree(q))
        anchor_nodes.append(best_qubit)

    # Statistics
    vis_chain_lens = [len(embedding[i]) for i in range(n_logical_visible)]
    n_logical_hidden = len(embedding) - n_logical_visible
    hid_chain_lens = [len(embedding[n_logical_visible + i]) for i in range(n_logical_hidden)]
    all_chain_qubits = set()
    for chain in embedding.values():
        all_chain_qubits.update(chain)
    free_qubits = set(G_zephyr.nodes()) - all_chain_qubits

    chain_info = {
        'n_anchors': len(anchor_nodes),
        'vis_chain_lens': vis_chain_lens,
        'hid_chain_lens': hid_chain_lens,
        'total_chain_qubits': len(all_chain_qubits),
        'free_qubits': len(free_qubits),
    }

    print(f"  Anchors selected: {len(anchor_nodes)}")
    print(f"  Visible chains: min={min(vis_chain_lens)} mean={np.mean(vis_chain_lens):.1f} "
          f"max={max(vis_chain_lens)}")
    print(f"  Hidden chains:  min={min(hid_chain_lens)} mean={np.mean(hid_chain_lens):.1f} "
          f"max={max(hid_chain_lens)}")
    print(f"  Qubits in chains: {len(all_chain_qubits)}, "
          f"free qubits: {len(free_qubits)}")

    return anchor_nodes, chain_info


# =========================================================================
# Anchor-to-subgrid spatial matching
# =========================================================================

def match_anchors_to_subgrid(G_zephyr, anchor_nodes, initial_pixels, grid_shape):
    """
    Match anchor nodes to initial sub-grid pixel positions using spatial
    proximity in the Zephyr layout.

    Args:
        G_zephyr: Zephyr graph (for layout positions)
        anchor_nodes: list of physical qubit IDs
        initial_pixels: list of (row, col) pixel positions
        grid_shape: (nrows, ncols)

    Returns:
        anchor_pixel_map: dict { qubit_id → (row, col) }
    """
    pos = get_zephyr_positions(G_zephyr)
    nrows, ncols = grid_shape

    # Map pixel (row, col) → layout (x, y)
    all_coords = np.array([pos[n] for n in sorted(G_zephyr.nodes())])
    xmin, xmax = all_coords[:, 0].min(), all_coords[:, 0].max()
    ymin, ymax = all_coords[:, 1].min(), all_coords[:, 1].max()

    def pixel_to_layout(r, c):
        lx = xmin + (c + 0.5) / ncols * (xmax - xmin)
        ly = ymax - (r + 0.5) / nrows * (ymax - ymin)
        return np.array([lx, ly])

    # Greedy nearest-neighbour matching
    used_anchors = set()
    anchor_pixel_map = {}

    for r, c in initial_pixels:
        target = pixel_to_layout(r, c)
        best_node = None
        best_dist = float('inf')
        for node in anchor_nodes:
            if node in used_anchors:
                continue
            d = np.sum((np.array(pos[node]) - target) ** 2)
            if d < best_dist:
                best_dist = d
                best_node = node
        anchor_pixel_map[best_node] = (r, c)
        used_anchors.add(best_node)

    return anchor_pixel_map


# =========================================================================
# Expansion: select additional visible nodes
# =========================================================================

def _multi_source_bfs_distances(G, sources):
    """
    Compute shortest-path distance from every node to the nearest source.
    Uses multi-source BFS (O(V + E)).

    Returns: dict { node → distance }
    """
    dist = {s: 0 for s in sources}
    queue = deque(sources)
    while queue:
        u = queue.popleft()
        for v in G.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def expand_visible_set(G_zephyr, anchor_nodes, num_needed,
                       add_node_criterion='graph_distance'):
    """
    Expand the visible set by selecting additional qubits from the pool.

    Args:
        G_zephyr: full Zephyr graph
        anchor_nodes: list of anchor visible node IDs
        num_needed: number of additional visible nodes to select
        add_node_criterion: "graph_distance" or "connectivity"

    Returns:
        expansion_nodes: list of newly selected visible qubit IDs
    """
    anchor_set = set(anchor_nodes)
    pool = [n for n in G_zephyr.nodes() if n not in anchor_set]

    if num_needed <= 0:
        return []
    if num_needed > len(pool):
        raise ValueError(
            f"Need {num_needed} expansion nodes but only {len(pool)} available "
            f"in the pool. Increase K or decrease image size."
        )

    if add_node_criterion == 'graph_distance':
        # Score = BFS distance to nearest anchor (lower = better)
        dist = _multi_source_bfs_distances(G_zephyr, list(anchor_set))
        # Sort pool by distance ascending, break ties by node degree descending
        pool_scored = sorted(pool, key=lambda n: (dist.get(n, 9999), -G_zephyr.degree(n)))
        expansion_nodes = pool_scored[:num_needed]

    elif add_node_criterion == 'connectivity':
        # Greedy: iteratively pick the node with the most non-visible neighbours
        visible_set = set(anchor_nodes)
        expansion_nodes = []
        pool_set = set(pool)

        for _ in range(num_needed):
            best_node = None
            best_score = -1
            for n in pool_set:
                # Count neighbours that are NOT visible (will be hidden)
                score = sum(1 for nb in G_zephyr.neighbors(n) if nb not in visible_set)
                if score > best_score or (score == best_score and
                                          G_zephyr.degree(n) > G_zephyr.degree(best_node)):
                    best_score = score
                    best_node = n
            expansion_nodes.append(best_node)
            visible_set.add(best_node)
            pool_set.remove(best_node)
    else:
        raise ValueError(f"Unknown add_node_criterion: {add_node_criterion!r}. "
                         f"Use 'graph_distance' or 'connectivity'.")

    print(f"  Expansion: selected {len(expansion_nodes)} additional visible nodes "
          f"(criterion={add_node_criterion!r})")

    # Report distance stats
    dist = _multi_source_bfs_distances(G_zephyr, list(anchor_set))
    exp_dists = [dist.get(n, -1) for n in expansion_nodes]
    print(f"  Expansion node distance to nearest anchor: "
          f"min={min(exp_dists)} mean={np.mean(exp_dists):.2f} max={max(exp_dists)}")

    return expansion_nodes


# =========================================================================
# Spatial pixel assignment for expansion nodes
# =========================================================================

def assign_expansion_pixels(G_zephyr, expansion_nodes, remaining_pixels, grid_shape):
    """
    Match expansion nodes to remaining pixel positions using spatial proximity
    in the Zephyr layout.

    Args:
        G_zephyr: Zephyr graph
        expansion_nodes: list of physical qubit IDs
        remaining_pixels: list of (row, col) positions
        grid_shape: (nrows, ncols)

    Returns:
        expansion_pixel_map: dict { qubit_id → (row, col) }
    """
    pos = get_zephyr_positions(G_zephyr)
    nrows, ncols = grid_shape

    all_coords = np.array([pos[n] for n in sorted(G_zephyr.nodes())])
    xmin, xmax = all_coords[:, 0].min(), all_coords[:, 0].max()
    ymin, ymax = all_coords[:, 1].min(), all_coords[:, 1].max()

    def pixel_to_layout(r, c):
        lx = xmin + (c + 0.5) / ncols * (xmax - xmin)
        ly = ymax - (r + 0.5) / nrows * (ymax - ymin)
        return np.array([lx, ly])

    # Build node position array for expansion nodes
    exp_coords = np.array([pos[n] for n in expansion_nodes])

    # Greedy matching: for each pixel, find the nearest unassigned expansion node
    used = set()
    expansion_pixel_map = {}

    for r, c in remaining_pixels:
        target = pixel_to_layout(r, c)
        best_node = None
        best_dist = float('inf')
        for idx, node in enumerate(expansion_nodes):
            if node in used:
                continue
            d = np.sum((exp_coords[idx] - target) ** 2)
            if d < best_dist:
                best_dist = d
                best_node = node
        if best_node is not None:
            expansion_pixel_map[best_node] = (r, c)
            used.add(best_node)

    return expansion_pixel_map


# =========================================================================
# Cache path helper (used by FULL-MNIST-hyperparam-search.py)
# =========================================================================

def get_arch4_cache_path(K, grid_shape, initial_grid_size, add_node_criterion,
                         data_dir='data'):
    """Get the cache file path for an ARCH4 assignment."""
    nrows, ncols = grid_shape
    return os.path.join(
        data_dir,
        f"arch4_assignment_K{K}_{nrows}x{ncols}_init{initial_grid_size}_{add_node_criterion}.json"
    )


# =========================================================================
# Main assignment function (the one imported by FULL-MNIST scripts)
# =========================================================================

def embedding_hybrid_assignment(G_zephyr, grid_shape, initial_grid_size=9,
                                add_node_criterion='graph_distance',
                                K=None,
                                embedding_timeout=60, embedding_tries=10,
                                data_dir='data'):
    """
    ARCH4: Embedding-based hybrid architecture.

    Embeds a small complete bipartite graph (RBM skeleton) into the Zephyr
    graph, breaks chains to get "anchor" visible nodes, then expands the
    visible set to cover all image pixels.

    Args:
        G_zephyr: Zephyr graph Z(K)
        grid_shape: (nrows, ncols), e.g. (28, 28)
        initial_grid_size: side length g of the initial sub-grid (g² core pixels)
        add_node_criterion: "graph_distance" or "connectivity"
        K: Zephyr parameter (used for cache filenames; auto-detected if None)
        embedding_timeout: seconds for minorminer per attempt
        embedding_tries: number of embedding attempts
        data_dir: directory for cached assignments (default: 'data')

    Returns:
        visible_in_pixel_order: list of node IDs, one per pixel, in row-major order
        hidden_nodes: list of remaining node IDs

        Returns (None, None) if embedding fails.
    """
    nrows, ncols = grid_shape
    num_visible = nrows * ncols  # total pixels we need to assign
    g = initial_grid_size
    n_core = g * g

    print(f"\n--- ARCH4: Embedding-Based Hybrid ---")
    print(f"  Image: {nrows}x{ncols} = {num_visible} pixels")
    print(f"  Initial grid: {g}x{g} = {n_core} core pixels")
    print(f"  Expansion criterion: {add_node_criterion}")

    # Auto-detect K from graph size if not provided
    if K is None:
        n = G_zephyr.number_of_nodes()
        # n = 16K(2K+1), solve for K
        for k_try in range(1, 30):
            if 16 * k_try * (2 * k_try + 1) == n:
                K = k_try
                break
        if K is None:
            raise ValueError(f"Cannot determine K from graph with {n} nodes")

    # ---- Phase 1: Compute initial sub-grid ----
    print("\nPhase 1: Computing initial sub-grid positions...")
    initial_pixels, remaining_pixels = compute_initial_subgrid(grid_shape, g)
    print(f"  Core pixels: {len(initial_pixels)}, Remaining: {len(remaining_pixels)}")

    # Visualize the sub-grid
    grid_vis = np.zeros(grid_shape, dtype=int)
    for r, c in initial_pixels:
        grid_vis[r, c] = 1
    print(f"  Sub-grid preview (1=initial, 0=expansion):")
    for row in grid_vis:
        print(f"    {''.join('i' if x else '.' for x in row)}")

    # ---- Phase 2: Embed and break chains ----
    print(f"\nPhase 2: Embedding K_{{{n_core},{n_core}}} into Z({K})...")

    # Try to load from cache first
    os.makedirs(data_dir, exist_ok=True)
    cache_path = get_arch4_cache_path(K, grid_shape, g, add_node_criterion,
                                      data_dir)

    if os.path.exists(cache_path):
        print(f"  Loading cached assignment from {cache_path}")
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        visible_in_pixel_order = cached['visible_in_pixel_order']
        hidden_nodes = cached['hidden_nodes']
        print(f"  Loaded: {len(visible_in_pixel_order)} visible, {len(hidden_nodes)} hidden")
        return visible_in_pixel_order, hidden_nodes

    # Find embedding
    embedding = find_bipartite_embedding(
        G_zephyr, n_core, n_core,
        timeout=embedding_timeout, tries=embedding_tries
    )
    if embedding is None:
        print(f"  ⚠️  Embedding failed. Try a smaller initial_grid_size or larger K.")
        return None, None

    # Break chains, get anchors
    print("\n  Breaking chains and selecting anchor nodes...")
    anchor_nodes, chain_info = break_chains_and_select_anchors(
        G_zephyr, embedding, n_core
    )

    # Match anchors to sub-grid pixels
    print("  Matching anchors to sub-grid pixel positions...")
    anchor_pixel_map = match_anchors_to_subgrid(
        G_zephyr, anchor_nodes, initial_pixels, grid_shape
    )

    # ---- Phase 3: Expand visible set ----
    num_expansion = num_visible - n_core
    print(f"\nPhase 3: Expanding visible set by {num_expansion} nodes...")
    expansion_nodes = expand_visible_set(
        G_zephyr, anchor_nodes, num_expansion,
        add_node_criterion=add_node_criterion
    )

    # Match expansion nodes to remaining pixels
    print("  Assigning expansion pixels spatially...")
    expansion_pixel_map = assign_expansion_pixels(
        G_zephyr, expansion_nodes, remaining_pixels, grid_shape
    )

    # ---- Combine into pixel-order list ----
    # Merge anchor and expansion pixel maps
    node_to_pixel_rc = {}
    node_to_pixel_rc.update(anchor_pixel_map)
    node_to_pixel_rc.update(expansion_pixel_map)

    # Build pixel_idx → node mapping
    pixel_to_node = {}
    for node, (r, c) in node_to_pixel_rc.items():
        pix_idx = r * ncols + c
        pixel_to_node[pix_idx] = node

    # Sanity checks
    assigned_pixels = set(pixel_to_node.keys())
    expected_pixels = set(range(num_visible))
    if assigned_pixels != expected_pixels:
        missing = expected_pixels - assigned_pixels
        extra = assigned_pixels - expected_pixels
        if missing:
            print(f"  ⚠️  {len(missing)} pixels unassigned! Filling with remaining nodes...")
            # Fallback: assign missing pixels to unused nodes
            all_visible_nodes = set(anchor_nodes) | set(expansion_nodes)
            unused_pool = [n for n in sorted(G_zephyr.nodes())
                           if n not in all_visible_nodes]
            for pix_idx in sorted(missing):
                if unused_pool:
                    fallback_node = unused_pool.pop(0)
                    pixel_to_node[pix_idx] = fallback_node
                    all_visible_nodes.add(fallback_node)
        if extra:
            print(f"  ⚠️  {len(extra)} extra pixel assignments (should not happen)")

    visible_in_pixel_order = [pixel_to_node[i] for i in range(num_visible)]
    visible_set = set(visible_in_pixel_order)
    hidden_nodes = [n for n in sorted(G_zephyr.nodes()) if n not in visible_set]

    print(f"\n  Final: {len(visible_in_pixel_order)} visible, {len(hidden_nodes)} hidden")

    # ---- Save to cache ----
    with open(cache_path, 'w') as f:
        json.dump({
            'visible_in_pixel_order': visible_in_pixel_order,
            'hidden_nodes': hidden_nodes,
            'K': K,
            'initial_grid_size': g,
            'add_node_criterion': add_node_criterion,
            'num_visible': num_visible,
            'grid_shape': list(grid_shape),
            'n_anchors': len(anchor_nodes),
            'n_expansion': len(expansion_nodes),
        }, f, indent=2)
    print(f"  Cached assignment to {cache_path}")

    return visible_in_pixel_order, hidden_nodes


# =========================================================================
# Standalone script
# =========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(ARCH_LABEL)
    print("=" * 60)

    #%% 1. Config

    K = 3  # Zephyr graph parameter (small scale for standalone test)
    grid_shape = GRID_SHAPE  # (12, 12) from bolmaqua
    num_visible = grid_shape[0] * grid_shape[1]

    # Initial grid: for 12x12 on Z(3), use 4x4 = 16 core pixels
    initial_grid_size = 4
    add_node_criterion = 'graph_distance'

    # Training hyperparameters
    lr = 1e-3
    weight_decay = 0.0001
    batch_size = 64
    epochs = 30
    k_steps = 15
    persistent_chains = True

    # Sampling
    num_samples = 9
    gibbs_burn_in = 1000
    sa_start_temp = 10.0
    sa_end_temp = 0.2
    sa_iterations = 16
    neal_num_sweeps = 20

    # Build tag
    train_method = "PCD" if persistent_chains else "CD"
    hparam_tag = (f"{train_method}_K{K}_lr{lr}_l2{weight_decay}_bs{batch_size}"
                  f"_ep{epochs}_k{k_steps}_g{initial_grid_size}_{add_node_criterion}")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Zephyr K={K}, grid_shape={grid_shape}, num_visible={num_visible}")
    print(f"Initial grid: {initial_grid_size}x{initial_grid_size} = {initial_grid_size**2}")
    print(f"Criterion: {add_node_criterion}")
    print(f"Training: lr={lr}, epochs={epochs}, k_steps={k_steps}, batch_size={batch_size}")
    print(f"Hparam tag: {hparam_tag}")

    #%% 2. Data Loading

    print("\nLoading MNIST data...")
    data = load_data(grid_shape)
    if data is None:
        raise RuntimeError("Failed to load data")

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Data samples: {len(data)}, visible units: {num_visible}")

    #%% 3. Architecture Construction

    print(f"\nGenerating Zephyr graph Z({K})...")
    t0 = time.time()
    G_zephyr = dnx.zephyr_graph(K)
    n_total = G_zephyr.number_of_nodes()
    n_edges = G_zephyr.number_of_edges()
    print(f"  Nodes: {n_total}, Edges: {n_edges}")
    print(f"  Degree: min={min(dict(G_zephyr.degree()).values())}, "
          f"max={max(dict(G_zephyr.degree()).values())}")

    result = embedding_hybrid_assignment(
        G_zephyr, grid_shape,
        initial_grid_size=initial_grid_size,
        add_node_criterion=add_node_criterion,
        K=K,
    )

    if result[0] is None:
        raise RuntimeError("Embedding failed — try smaller initial_grid_size or larger K")

    visible_in_pixel_order, hidden_nodes = result
    num_hidden = len(hidden_nodes)

    print(f"\nRelabeling graph (visible=0..{num_visible-1}, "
          f"hidden={num_visible}..{num_visible+num_hidden-1})...")
    G_relabeled, mapping = relabel_visible_first(G_zephyr, visible_in_pixel_order)

    # Build node labels
    node_labels = {}
    for i in range(num_visible):
        node_labels[i] = 'visible'
    for i in range(num_visible, num_visible + num_hidden):
        node_labels[i] = 'hidden'

    t_arch = time.time() - t0
    print(f"Architecture construction time: {t_arch:.1f}s")

    #%% 4. Architecture Analysis

    visible_relabeled = list(range(num_visible))
    hidden_relabeled = list(range(num_visible, num_visible + num_hidden))
    stats = analyze_architecture(G_relabeled, visible_relabeled, hidden_relabeled, ARCH_LABEL)

    #%% 5. Model Initialization

    print("Initializing CustomBoltzmannMachine...")
    model = graph_to_bm(G_relabeled, node_labels)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model on {device}, total parameters: {total_params:,}")

    #%% 6. Training (PCD)

    print(f"\nStarting {train_method} Training (lr={lr}, epochs={epochs}, k={k_steps})...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    training_history = train_boltzmann_machine_pcd(
        model,
        loader,
        optimizer,
        num_epochs=epochs,
        k_steps=k_steps,
        batch_size=batch_size,
        step_size=lr,
        persistent=persistent_chains,
        train_data=data,
        eval_every=5,
    )

    #%% 6a. Save model and training history

    model_path = os.path.join(data_dir, f"arch4_{ARCH_NAME}_{hparam_tag}_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'arch_name': ARCH_NAME,
        'arch_label': ARCH_LABEL,
        'hyperparams': {
            'K': K, 'grid_shape': grid_shape, 'lr': lr, 'weight_decay': weight_decay,
            'l2_reg': weight_decay,
            'batch_size': batch_size, 'epochs': epochs, 'k_steps': k_steps,
            'initial_grid_size': initial_grid_size,
            'add_node_criterion': add_node_criterion,
            'num_visible': num_visible, 'num_hidden': num_hidden,
        },
        'graph_edges': list(G_relabeled.edges()),
        'node_labels': node_labels,
    }, model_path)
    print(f"Saved model to {model_path}")

    #%% 6b. Plot training metrics

    if training_history is not None and 'pcd_loss' in training_history:
        epochs_range = range(1, len(training_history['pcd_loss']) + 1)
        has_train_recon = len(training_history.get('train_recon_mse', [])) > 0
        num_plots = 2 if has_train_recon else 1

        fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]

        ax1 = axes[0]
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("PCD Loss", color='tab:blue')
        ax1.plot(epochs_range, training_history['pcd_loss'],
                 marker='o', linewidth=2, color='tab:blue', label='PCD Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        if 'pll' in training_history and len(training_history['pll']) > 0:
            ax1b = ax1.twinx()
            ax1b.set_ylabel("Pseudo Log-Likelihood", color='tab:red')
            ax1b.plot(epochs_range, training_history['pll'],
                      marker='s', linewidth=2, color='tab:red', label='PLL')
            ax1b.tick_params(axis='y', labelcolor='tab:red')
        ax1.set_title("PCD Loss & PLL")
        ax1.grid(True, alpha=0.3)

        if has_train_recon:
            ax2 = axes[1]
            eval_every = 5
            n_recon = len(training_history['train_recon_mse'])
            recon_epochs = [e for e in range(1, len(training_history['pcd_loss']) + 1)
                            if e % eval_every == 0 or e == len(training_history['pcd_loss'])]
            recon_epochs = recon_epochs[:n_recon]

            ax2.plot(recon_epochs, training_history['train_recon_mse'],
                     marker='o', linewidth=2, color='tab:green', label='MSE')
            ax2.plot(recon_epochs, training_history['train_recon_bce'],
                     marker='^', linewidth=2, color='tab:orange', label='BCE')
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Reconstruction Loss")
            ax2.set_title("Train Reconstruction Metrics")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')

            ax2b = ax2.twinx()
            ax2b.plot(recon_epochs, training_history['train_recon_acc'],
                      marker='D', linewidth=2, color='tab:purple', label='Accuracy')
            ax2b.set_ylabel("Accuracy", color='tab:purple')
            ax2b.tick_params(axis='y', labelcolor='tab:purple')
            ax2b.legend(loc='upper right')

        fig.suptitle(f"{ARCH_LABEL} — Training Metrics (Hidden={num_hidden})")
        fig.tight_layout()
        fig.savefig(f"arch4_{ARCH_NAME}_training.png")
        fig.savefig(os.path.join(data_dir, f"arch4_{ARCH_NAME}_{hparam_tag}_training.png"))
        print(f"Saved training plot to arch4_{ARCH_NAME}_training.png")
        plt.show()

    #%% 7. Sampling & Visualization (Gibbs)

    print(f"\nGenerating {num_samples} Gibbs samples (burn-in={gibbs_burn_in})...")
    samples = sample_from_bm(model, num_samples=num_samples, burn_in_steps=gibbs_burn_in,
                             method='gibbs')
    samples_np = samples.cpu().detach().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle(f"{ARCH_LABEL} — Gibbs Samples (Epochs={epochs}, Hidden={num_hidden})")
    for i, ax in enumerate(axes.flat):
        if i < len(samples_np):
            ax.imshow(samples_np[i].reshape(grid_shape), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"arch4_{ARCH_NAME}_samples_gibbs.png")
    plt.savefig(os.path.join(data_dir, f"arch4_{ARCH_NAME}_{hparam_tag}_samples_gibbs.png"))
    print(f"Saved Gibbs samples to arch4_{ARCH_NAME}_samples_gibbs.png")
    plt.show()

    #%% 8. Sampling & Visualization (Simulated Annealing)

    print(f"\nGenerating {num_samples} SA samples...")
    sa_samples = BM_SimAnn_Sampler(
        model=model,
        start_temp=sa_start_temp,
        end_temp=sa_end_temp,
        max_iterations=sa_iterations,
        num_samples=num_samples,
        track_best=True,
        verbose=True,
    )
    sa_np = sa_samples.cpu().detach().numpy()

    fig_sa, axes_sa = plt.subplots(3, 3, figsize=(8, 8))
    fig_sa.suptitle(f"{ARCH_LABEL} — SA Samples (Epochs={epochs}, Hidden={num_hidden})")
    for i, ax in enumerate(axes_sa.flat):
        if i < len(sa_np):
            ax.imshow(sa_np[i].reshape(grid_shape), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"arch4_{ARCH_NAME}_samples_sa.png")
    plt.savefig(os.path.join(data_dir, f"arch4_{ARCH_NAME}_{hparam_tag}_samples_sa.png"))
    print(f"Saved SA samples to arch4_{ARCH_NAME}_samples_sa.png")
    plt.show()

    #%% 9. Sampling & Visualization (Neal / D-Wave SA)

    print(f"\nGenerating {num_samples} Neal SA samples (sweeps={neal_num_sweeps})...")
    neal_samples = BM_Neal_Sampler(
        model=model,
        num_samples=num_samples,
        num_sweeps=neal_num_sweeps,
        verbose=True,
    )
    neal_np = neal_samples.cpu().detach().numpy()

    fig_neal, axes_neal = plt.subplots(3, 3, figsize=(8, 8))
    fig_neal.suptitle(f"{ARCH_LABEL} — Neal SA Samples (Epochs={epochs}, Hidden={num_hidden})")
    for i, ax in enumerate(axes_neal.flat):
        if i < len(neal_np):
            ax.imshow(neal_np[i].reshape(grid_shape), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"arch4_{ARCH_NAME}_samples_neal.png")
    plt.savefig(os.path.join(data_dir, f"arch4_{ARCH_NAME}_{hparam_tag}_samples_neal.png"))
    print(f"Saved Neal SA samples to arch4_{ARCH_NAME}_samples_neal.png")
    plt.show()

    print(f"\n{ARCH_LABEL} — Complete.")
