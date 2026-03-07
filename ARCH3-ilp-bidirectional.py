'''
ARCH3: ILP-Optimized Assignment with Bidirectional VH Objective

Strategy: Formulate the visible/hidden node assignment as an Integer Linear
Program (ILP) that directly maximizes a learning-quality proxy:
    maximize   alpha * t_V + beta * t_H - gamma * VV_edges
where:
    t_V = min VH-degree across all VISIBLE nodes  (every pixel sees hidden units)
    t_H = min VH-degree across all HIDDEN nodes   (every hidden sees pixels)
    VV_edges = count of visible-visible edges      (penalty: waste of edges)

This is a richer objective than the existing SCIP formulation in the
codebase (which only maximizes t_H). By optimizing bidirectionally,
we ensure both encoding (V→H) and decoding (H→V) are well-supported.

Uses the spectral bisection (ARCH1) solution as a warm start for SCIP.
Falls back to a greedy refinement approach if pyscipopt is unavailable.
'''

#%% defs

import os
import time
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import dwave_networkx as dnx

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

from optihelper import visualize_node_assignment_on_zephyr

# Check if SCIP is available
try:
    from pyscipopt import Model as SCIPModel, quicksum
    HAS_SCIP = True
except ImportError:
    HAS_SCIP = False
    print("Warning: pyscipopt not installed. Will use greedy fallback.")

ARCH_NAME = "ilp"
ARCH_LABEL = "ARCH3: ILP Bidirectional"


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


def assign_pixel_order_spatial(G, visible_set, grid_shape):
    """Assign pixel indices to visible nodes based on spatial proximity."""
    pos = get_zephyr_positions(G)
    visible_list = list(visible_set)
    nrows, ncols = grid_shape

    all_coords = np.array([pos[n] for n in sorted(G.nodes())])
    xmin, xmax = all_coords[:, 0].min(), all_coords[:, 0].max()
    ymin, ymax = all_coords[:, 1].min(), all_coords[:, 1].max()

    grid_x = np.linspace(xmin, xmax, ncols)
    grid_y = np.linspace(ymax, ymin, nrows)

    used = set()
    pixel_to_node = {}
    for r in range(nrows):
        for c in range(ncols):
            pix_idx = r * ncols + c
            target = np.array([grid_x[c], grid_y[r]])
            best_dist = float('inf')
            best_node = None
            for node in visible_list:
                if node in used:
                    continue
                d = np.sum((np.array(pos[node]) - target) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_node = node
            pixel_to_node[pix_idx] = best_node
            used.add(best_node)

    return [pixel_to_node[i] for i in range(nrows * ncols)]


def spectral_warm_start(G, num_visible):
    """Compute a Fiedler-based warm start partition for the ILP."""
    nodes = sorted(G.nodes())
    L = nx.laplacian_matrix(G).toarray().astype(float)
    _, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]

    sorted_indices = np.argsort(fiedler)
    vis_low  = set(nodes[i] for i in sorted_indices[:num_visible])
    vis_high = set(nodes[i] for i in sorted_indices[-num_visible:])

    def count_vh(vis):
        return sum(1 for u, v in G.edges() if (u in vis) != (v in vis))

    if count_vh(vis_low) >= count_vh(vis_high):
        return vis_low
    return vis_high


def ilp_bidirectional_assignment(G, num_visible, alpha=1.0, beta=1.0, gamma=0.01,
                                  time_limit=300, warm_start_set=None):
    """
    ILP formulation:
        max   alpha * t_V + beta * t_H - gamma * sum(y_{ij})

    Variables:
        x_i ∈ {0,1}  — 1 if node i is visible
        y_{ij} ∈ {0,1} — linearization of x_i * x_j for each edge (VV indicator)
        t_V ∈ Z≥0     — min VH-degree across visible nodes
        t_H ∈ Z≥0     — min VH-degree across hidden nodes

    Constraints:
        sum(x_i) = num_visible
        y_{ij} <= x_i, y_{ij} <= x_j, y_{ij} >= x_i + x_j - 1   (linearization)
        t_V <= sum_{j∈N(i)}(1-x_j) + M*(1-x_i)  ∀i   (min-VH for visible)
        t_H <= sum_{j∈N(i)}(x_j)   + M*(x_i)    ∀i   (min-VH for hidden)
    """
    if not HAS_SCIP:
        raise ImportError("pyscipopt required for ILP; use greedy fallback instead")

    nodes = sorted(G.nodes())
    edges = list(G.edges())
    n = len(nodes)
    adj = {i: list(G.neighbors(i)) for i in nodes}
    max_deg = max(G.degree(i) for i in nodes)
    M = max_deg + 1

    m = SCIPModel("bidirectional_vh")
    m.setParam("limits/time", time_limit)
    m.setParam("display/verblevel", 3)

    # --- Variables ---
    x = {i: m.addVar(vtype="B", name=f"x_{i}") for i in nodes}
    # y_{ij} linearizes x_i * x_j. The three linearization constraints guarantee
    # y = x_i * x_j at any integer solution, so y can be continuous [0,1] —
    # no need to branch on these ~11K variables.
    y = {(i, j): m.addVar(vtype="C", lb=0.0, ub=1.0, name=f"y_{i}_{j}") for (i, j) in edges}
    t_V = m.addVar(vtype="I", lb=0, ub=max_deg, name="t_V")
    t_H = m.addVar(vtype="I", lb=0, ub=max_deg, name="t_H")

    # --- Constraints ---
    # 1. Exactly num_visible visible nodes
    m.addCons(quicksum(x[i] for i in nodes) == num_visible, "visible_count")

    # 2. Linearize y_{ij} = x_i * x_j
    for (i, j) in edges:
        m.addCons(y[(i, j)] <= x[i], f"y_ub_i_{i}_{j}")
        m.addCons(y[(i, j)] <= x[j], f"y_ub_j_{i}_{j}")
        m.addCons(y[(i, j)] >= x[i] + x[j] - 1, f"y_lb_{i}_{j}")

    # 3. Maximin for visible: t_V <= hidden_degree(i) + M*(1-x_i) ∀i
    #    hidden_degree(i) = sum_{j∈N(i)}(1 - x_j)
    for i in nodes:
        hidden_deg = quicksum(1 - x[j] for j in adj[i])
        m.addCons(t_V <= hidden_deg + M * (1 - x[i]), f"t_V_{i}")

    # 4. Maximin for hidden: t_H <= visible_degree(i) + M*x_i ∀i
    #    visible_degree(i) = sum_{j∈N(i)} x_j
    for i in nodes:
        vis_deg = quicksum(x[j] for j in adj[i])
        m.addCons(t_H <= vis_deg + M * x[i], f"t_H_{i}")

    # --- Objective ---
    vv_total = quicksum(y[(i, j)] for (i, j) in edges)
    m.setObjective(alpha * t_V + beta * t_H - gamma * vv_total, "maximize")

    # --- Warm start ---
    if warm_start_set is not None:
        sol = m.createSol()
        for i in nodes:
            m.setSolVal(sol, x[i], 1.0 if i in warm_start_set else 0.0)
        # Set y values consistent with warm start x
        for (i, j) in edges:
            val = 1.0 if (i in warm_start_set and j in warm_start_set) else 0.0
            m.setSolVal(sol, y[(i, j)], val)
        # Set t_V, t_H from warm start
        ws_vis = warm_start_set
        ws_hid = set(nodes) - ws_vis
        min_tv = min(sum(1 for nb in adj[i] if nb not in ws_vis) for i in ws_vis) if ws_vis else 0
        min_th = min(sum(1 for nb in adj[j] if nb in ws_vis) for j in ws_hid) if ws_hid else 0
        m.setSolVal(sol, t_V, float(min_tv))
        m.setSolVal(sol, t_H, float(min_th))
        try:
            accepted = m.addSol(sol)
            print(f"  Warm start {'accepted' if accepted else 'rejected'} "
                  f"(t_V={min_tv}, t_H={min_th})")
        except Exception as e:
            print(f"  Warm start failed: {e}")

    print(f"\n  Solving ILP ({n} nodes, {len(edges)} edges, "
          f"{n + len(edges) + 2} variables, time_limit={time_limit}s)...")
    m.optimize()

    if m.getNSols() == 0:
        print("  ERROR: No solution found!")
        return None, None

    sol = m.getBestSol()
    visible_set = set()
    for i in nodes:
        if m.getSolVal(sol, x[i]) > 0.5:
            visible_set.add(i)

    t_V_val = m.getSolVal(sol, t_V)
    t_H_val = m.getSolVal(sol, t_H)
    vv_val = sum(1 for (i, j) in edges
                 if m.getSolVal(sol, x[i]) > 0.5 and m.getSolVal(sol, x[j]) > 0.5)
    obj_val = m.getSolObjVal(sol)
    gap = m.getGap()

    print(f"\n  ILP Solution:")
    print(f"    t_V (min VH-deg visible) = {t_V_val:.0f}")
    print(f"    t_H (min VH-deg hidden)  = {t_H_val:.0f}")
    print(f"    VV edges                 = {vv_val}")
    print(f"    Objective                = {obj_val:.3f}")
    print(f"    Optimality gap           = {100*gap:.2f}%")

    return visible_set, {'t_V': t_V_val, 't_H': t_H_val, 'vv': vv_val,
                         'obj': obj_val, 'gap': gap}


def greedy_bidirectional_assignment(G, num_visible, gamma=0.01, max_iters=1000):
    """
    Greedy fallback: start from Fiedler partition, then do random swaps
    optimizing VH_edges - gamma * VV_edges as a proxy for the ILP objective.

    After convergence, reports the bidirectional min VH-degree statistics.
    """
    print("  Using greedy fallback (SCIP not available)...")
    visible_set = spectral_warm_start(G, num_visible)
    nodes = sorted(G.nodes())

    def count_edges(vis):
        vh = vv = 0
        for u, v in G.edges():
            uv, vv_ = u in vis, v in vis
            if uv != vv_:   vh += 1
            elif uv and vv_: vv += 1
        return vh, vv

    vh0, vv0 = count_edges(visible_set)
    best_obj = vh0 - gamma * vv0
    swaps = 0

    for _ in range(max_iters):
        v_list = list(visible_set)
        h_list = [nd for nd in nodes if nd not in visible_set]
        v_node = v_list[np.random.randint(len(v_list))]
        h_node = h_list[np.random.randint(len(h_list))]

        delta_vh = delta_vv = 0
        for nb in G.neighbors(v_node):
            if nb == h_node: continue
            if nb in visible_set:
                delta_vh += 1; delta_vv -= 1
            else:
                delta_vh -= 1
        for nb in G.neighbors(h_node):
            if nb == v_node: continue
            if nb in visible_set:
                delta_vh -= 1; delta_vv += 1
            else:
                delta_vh += 1

        change = delta_vh - gamma * delta_vv
        if change > 0:
            visible_set.remove(v_node)
            visible_set.add(h_node)
            best_obj += change
            swaps += 1

    # Phase 2: targeted fixing of worst-case visible nodes
    for _ in range(max_iters // 5):
        v_list = list(visible_set)
        h_list = [nd for nd in nodes if nd not in visible_set]

        # Find visible node with lowest hidden-degree
        worst_v = min(v_list,
                      key=lambda n: sum(1 for nb in G.neighbors(n) if nb not in visible_set))
        worst_deg = sum(1 for nb in G.neighbors(worst_v) if nb not in visible_set)
        if worst_deg >= 2:
            break  # Acceptable

        # Try swapping with random hidden nodes
        h_candidates = np.random.choice(h_list, min(50, len(h_list)), replace=False)
        best_swap = None
        best_new_min = worst_deg
        for h_node in h_candidates:
            # After swap: worst_v becomes hidden, h_node becomes visible
            # Check h_node's would-be hidden-degree:
            h_node_hid_deg = sum(1 for nb in G.neighbors(h_node)
                                 if nb not in visible_set and nb != worst_v)
            # Also need to check that no other visible node's min gets worse
            if h_node_hid_deg > best_new_min:
                best_new_min = h_node_hid_deg
                best_swap = h_node

        if best_swap is not None and best_new_min > worst_deg:
            visible_set.remove(worst_v)
            visible_set.add(best_swap)
            swaps += 1

    vh_f, vv_f = count_edges(visible_set)
    print(f"  Greedy: {swaps} swaps, VH={vh0}→{vh_f}, VV={vv0}→{vv_f}")

    # Report bidirectional min
    v_list = list(visible_set)
    h_list = [nd for nd in nodes if nd not in visible_set]
    min_tv = min(sum(1 for nb in G.neighbors(v) if nb not in visible_set) for v in v_list)
    min_th = min(sum(1 for nb in G.neighbors(h) if nb in visible_set) for h in h_list)
    print(f"  Bidirectional min: t_V={min_tv}, t_H={min_th}")

    return visible_set



if __name__ == '__main__':
    print("=" * 60)
    print(ARCH_LABEL)
    print(f"SCIP available: {HAS_SCIP}")
    print("=" * 60)

    #%% 1. Config

    K = 3  # Zephyr graph parameter
    grid_shape = GRID_SHAPE  # (12, 12) from bolmaqua
    num_visible = grid_shape[0] * grid_shape[1]

    # Training hyperparameters (matching test-RBM-as-custom-BM.py)
    lr = 1e-3
    weight_decay = 0.00001
    batch_size = 64
    epochs = 50
    k_steps = 15
    persistent_chains = True

    # Sampling
    num_samples = 9
    gibbs_burn_in = 1000
    sa_start_temp = 10.0
    sa_end_temp = 0.2
    sa_iterations = 16
    neal_num_sweeps = 20

    # Strategy-specific: ILP objective weights
    ilp_alpha = 1.0    # weight for min VH-degree of visible nodes
    ilp_beta  = 1.0    # weight for min VH-degree of hidden nodes
    ilp_gamma = 0.01   # penalty per VV edge
    ilp_time_limit = 300  # seconds (5 minutes; K=3 should solve much faster)

    # Build hyperparam tag for filenames
    train_method = "PCD" if persistent_chains else "CD"
    hparam_tag = f"{train_method}_K{K}_lr{lr}_l2{weight_decay}_bs{batch_size}_ep{epochs}_k{k_steps}_a{ilp_alpha}_b{ilp_beta}_g{ilp_gamma}"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Zephyr K={K}, grid_shape={grid_shape}, num_visible={num_visible}")
    print(f"ILP weights: alpha={ilp_alpha}, beta={ilp_beta}, gamma={ilp_gamma}")
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

    # Step 1: Compute spectral warm start
    print("\nStep 1: Computing spectral warm start...")
    warm_start = spectral_warm_start(G_zephyr, num_visible)
    ws_vh = sum(1 for u, v in G_zephyr.edges() if (u in warm_start) != (v in warm_start))
    print(f"  Warm start VH edges: {ws_vh}")

    # Step 2: Run ILP or greedy
    if HAS_SCIP:
        print("\nStep 2: Solving ILP with SCIP...")
        visible_set, ilp_stats = ilp_bidirectional_assignment(
            G_zephyr, num_visible,
            alpha=ilp_alpha, beta=ilp_beta, gamma=ilp_gamma,
            time_limit=ilp_time_limit,
            warm_start_set=warm_start,
        )
        if visible_set is None:
            print("  ILP failed — falling back to greedy")
            visible_set = greedy_bidirectional_assignment(
                G_zephyr, num_visible, gamma=ilp_gamma)
    else:
        print("\nStep 2: Greedy bidirectional optimization (SCIP not available)...")
        visible_set = greedy_bidirectional_assignment(
            G_zephyr, num_visible, gamma=ilp_gamma)

    # Step 3: Assign pixel positions spatially
    print("\nStep 3: Spatial pixel assignment...")
    visible_in_pixel_order = assign_pixel_order_spatial(G_zephyr, visible_set, grid_shape)
    hidden_nodes = [n for n in sorted(G_zephyr.nodes()) if n not in visible_set]
    num_hidden = len(hidden_nodes)

    # Step 4: Relabel graph
    print(f"\nStep 4: Relabeling graph (visible=0..{num_visible-1}, "
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

    visualize_node_assignment_on_zephyr(G_relabeled, visible_relabeled, hidden_relabeled, 
                                    title="Final Assignment (Red=Visible, Blue=Hidden)")

    #%% 5. Model Initialization

    print("Initializing CustomBoltzmannMachine...")
    model = graph_to_bm(G_relabeled, node_labels)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model on {device}, total parameters: {total_params:,}")

    #%% 6. Training (PCD)

    print(f"\nStarting PCD Training (lr={lr}, epochs={epochs}, k={k_steps})...")
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

    model_path = os.path.join(data_dir, f"arch3_{ARCH_NAME}_{hparam_tag}_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'arch_name': ARCH_NAME,
        'arch_label': ARCH_LABEL,
        'hyperparams': {
            'K': K, 'grid_shape': grid_shape, 'lr': lr, 'weight_decay': weight_decay, 'l2_reg': weight_decay,
            'batch_size': batch_size, 'epochs': epochs, 'k_steps': k_steps,
            'ilp_alpha': ilp_alpha, 'ilp_beta': ilp_beta, 'ilp_gamma': ilp_gamma,
            'ilp_time_limit': ilp_time_limit,
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

        # Panel 1: PCD Loss & PLL
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

        # Panel 2: Reconstruction metrics
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

        method_str = "ILP" if HAS_SCIP else "Greedy"
        fig.suptitle(f"{ARCH_LABEL} ({method_str}) — Training Metrics (Hidden={num_hidden})")
        fig.tight_layout()
        fig.savefig(f"arch3_{ARCH_NAME}_training.png")
        fig.savefig(os.path.join(data_dir, f"arch3_{ARCH_NAME}_{hparam_tag}_training.png"))
        print(f"Saved training plot to arch3_{ARCH_NAME}_training.png")
        plt.show()

    #%% 7. Sampling & Visualization (Gibbs)

    print(f"\nGenerating {num_samples} Gibbs samples (burn-in={gibbs_burn_in})...")
    samples = sample_from_bm(model, num_samples=num_samples, burn_in_steps=gibbs_burn_in, method='gibbs')
    samples_np = samples.cpu().detach().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle(f"{ARCH_LABEL} — Gibbs Samples (Epochs={epochs}, Hidden={num_hidden})")
    for i, ax in enumerate(axes.flat):
        if i < len(samples_np):
            ax.imshow(samples_np[i].reshape(grid_shape), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"arch3_{ARCH_NAME}_samples_gibbs.png")
    plt.savefig(os.path.join(data_dir, f"arch3_{ARCH_NAME}_{hparam_tag}_samples_gibbs.png"))
    print(f"Saved Gibbs samples to arch3_{ARCH_NAME}_samples_gibbs.png")
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
    plt.savefig(f"arch3_{ARCH_NAME}_samples_sa.png")
    plt.savefig(os.path.join(data_dir, f"arch3_{ARCH_NAME}_{hparam_tag}_samples_sa.png"))
    print(f"Saved SA samples to arch3_{ARCH_NAME}_samples_sa.png")
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
    plt.savefig(f"arch3_{ARCH_NAME}_samples_neal.png")
    plt.savefig(os.path.join(data_dir, f"arch3_{ARCH_NAME}_{hparam_tag}_samples_neal.png"))
    print(f"Saved Neal SA samples to arch3_{ARCH_NAME}_samples_neal.png")
    plt.show()

    print(f"\n{ARCH_LABEL} — Complete.")
