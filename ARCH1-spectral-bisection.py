'''
ARCH1: Spectral Bisection with Receptive-Field Maximization

Strategy: Use the Fiedler vector (2nd eigenvector of the graph Laplacian)
to partition the Zephyr graph into visible/hidden sets that approximately
maximize the visible-hidden edge cut. Then assign pixel indices to visible
nodes using spatial locality from the Zephyr layout. Finally, apply local
swap refinement to further improve the VH cut.

Rationale: The Fiedler vector identifies the graph's natural "two sides"
with maximum cross-connections, approximating a max-cut. This makes the
resulting BM most similar to an RBM (dense VH connectivity), while the
non-bipartite edges (VV, HH) provide additional learning capacity.
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

ARCH_NAME = "spectral"
ARCH_LABEL = "ARCH1: Spectral Bisection"


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
    """
    Given a set of visible nodes, assign pixel indices based on
    spatial proximity in the Zephyr layout (row-major order).
    Returns: list of node IDs in pixel order (pixel 0 → node, pixel 1 → node, ...).
    """
    pos = get_zephyr_positions(G)
    visible_list = list(visible_set)
    nrows, ncols = grid_shape

    all_coords = np.array([pos[n] for n in sorted(G.nodes())])
    xmin, xmax = all_coords[:, 0].min(), all_coords[:, 0].max()
    ymin, ymax = all_coords[:, 1].min(), all_coords[:, 1].max()

    grid_x = np.linspace(xmin, xmax, ncols)
    grid_y = np.linspace(ymax, ymin, nrows)  # flip y so row 0 = top

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


def spectral_bisection_partition(G, num_visible):
    """
    Use the Fiedler vector to partition nodes into visible/hidden
    sets that approximately maximize the VH edge cut.
    Returns: visible_set (set of node IDs)
    """
    nodes = sorted(G.nodes())
    n = len(nodes)

    # Dense Laplacian eigendecomposition (fast for n < ~2000)
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler = eigenvectors[:, 1]  # 2nd eigenvector (sorted ascending)

    print(f"  Fiedler eigenvalue (algebraic connectivity): {eigenvalues[1]:.4f}")

    # Sort nodes by Fiedler value
    node_fiedler = [(nodes[i], fiedler[i]) for i in range(n)]
    node_fiedler.sort(key=lambda x: x[1])

    # Try both orientations: bottom num_visible vs top num_visible
    vis_low  = set(nf[0] for nf in node_fiedler[:num_visible])
    vis_high = set(nf[0] for nf in node_fiedler[-num_visible:])

    def count_vh(vis):
        return sum(1 for u, v in G.edges() if (u in vis) != (v in vis))

    vh_low  = count_vh(vis_low)
    vh_high = count_vh(vis_high)
    print(f"  Fiedler partition VH edges: low-end={vh_low}, high-end={vh_high}")

    visible_set = vis_low if vh_low >= vh_high else vis_high
    print(f"  Selected: {'low-end' if vh_low >= vh_high else 'high-end'} ({max(vh_low, vh_high)} VH edges)")
    return visible_set


def refine_partition_swaps(G, visible_set, num_visible, max_iters=500, gamma=0.01):
    """
    Local refinement: random visible/hidden swaps to improve
    VH_edges - gamma * VV_edges.
    """
    nodes = sorted(G.nodes())
    visible_set = set(visible_set)  # copy

    def count_edges(vis):
        vh = vv = 0
        for u, v in G.edges():
            uv, vv_ = u in vis, v in vis
            if uv != vv_:   vh += 1
            elif uv and vv_: vv += 1
        return vh, vv

    vh0, vv0 = count_edges(visible_set)
    best_obj = vh0 - gamma * vv0
    swaps_accepted = 0

    for iteration in range(max_iters):
        v_list = list(visible_set)
        h_list = [nd for nd in nodes if nd not in visible_set]

        vi = np.random.randint(len(v_list))
        hi = np.random.randint(len(h_list))
        v_node, h_node = v_list[vi], h_list[hi]

        # Incremental change in VH and VV from this swap
        delta_vh = delta_vv = 0
        for nb in G.neighbors(v_node):
            if nb == h_node: continue
            if nb in visible_set:
                delta_vh += 1; delta_vv -= 1   # VV → VH
            else:
                delta_vh -= 1                  # VH → HH
        for nb in G.neighbors(h_node):
            if nb == v_node: continue
            if nb in visible_set:
                delta_vh -= 1; delta_vv += 1   # VH → VV
            else:
                delta_vh += 1                  # HH → VH

        change = delta_vh - gamma * delta_vv
        if change > 0:
            visible_set.remove(v_node)
            visible_set.add(h_node)
            best_obj += change
            swaps_accepted += 1

    vh_final, vv_final = count_edges(visible_set)
    print(f"  Refinement: {swaps_accepted} swaps accepted over {max_iters} iterations")
    print(f"  VH edges: {vh0} → {vh_final}  |  VV edges: {vv0} → {vv_final}")
    return visible_set



if __name__ == '__main__':
    print("=" * 60)
    print(ARCH_LABEL)
    print("=" * 60)

    #%% 1. Config

    K = 3  # Zephyr graph parameter
    grid_shape = GRID_SHAPE  # (12, 12) from bolmaqua
    num_visible = grid_shape[0] * grid_shape[1]

    # Training hyperparameters (matching test-RBM-as-custom-BM.py)
    lr = 5e-3
    weight_decay = 0.001
    batch_size = 64
    epochs = 10
    k_steps = 5
    persistent_chains = False

    # Sampling
    num_samples = 9
    gibbs_burn_in = 1000
    sa_start_temp = 10.0
    sa_end_temp = 0.2
    sa_iterations = 16
    neal_num_sweeps = 20

    # Strategy-specific
    refinement_iters = 500
    vv_penalty = 0.01  # gamma for VV edge penalty in refinement

    # Build hyperparam tag for filenames
    train_method = "PCD" if persistent_chains else "CD"
    hparam_tag = f"{train_method}_K{K}_lr{lr}_l2{weight_decay}_bs{batch_size}_ep{epochs}_k{k_steps}_refine{refinement_iters}_gamma{vv_penalty}"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Zephyr K={K}, grid_shape={grid_shape}, num_visible={num_visible}")
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

    print("\nStep 1: Spectral bisection partition...")
    visible_set = spectral_bisection_partition(G_zephyr, num_visible)

    print("\nStep 2: Local swap refinement...")
    visible_set = refine_partition_swaps(G_zephyr, visible_set, num_visible,
                                          max_iters=refinement_iters, gamma=vv_penalty)

    print("\nStep 3: Spatial pixel assignment...")
    visible_in_pixel_order = assign_pixel_order_spatial(G_zephyr, visible_set, grid_shape)
    hidden_nodes = [n for n in sorted(G_zephyr.nodes()) if n not in visible_set]
    num_hidden = len(hidden_nodes)

    print(f"\nStep 4: Relabeling graph (visible=0..{num_visible-1}, hidden={num_visible}..{num_visible+num_hidden-1})...")
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

    model_path = os.path.join(data_dir, f"arch1_{ARCH_NAME}_{hparam_tag}_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'arch_name': ARCH_NAME,
        'arch_label': ARCH_LABEL,
        'hyperparams': {
            'K': K, 'grid_shape': grid_shape, 'lr': lr, 'weight_decay': weight_decay, 'l2_reg': weight_decay,
            'batch_size': batch_size, 'epochs': epochs, 'k_steps': k_steps,
            'refinement_iters': refinement_iters, 'vv_penalty': vv_penalty,
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

        fig.suptitle(f"{ARCH_LABEL} — Training Metrics (Hidden={num_hidden})")
        fig.tight_layout()
        fig.savefig(f"arch1_{ARCH_NAME}_training.png")
        fig.savefig(os.path.join(data_dir, f"arch1_{ARCH_NAME}_{hparam_tag}_training.png"))
        print(f"Saved training plot to arch1_{ARCH_NAME}_training.png")
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
    plt.savefig(f"arch1_{ARCH_NAME}_samples_gibbs.png")
    plt.savefig(os.path.join(data_dir, f"arch1_{ARCH_NAME}_{hparam_tag}_samples_gibbs.png"))
    print(f"Saved Gibbs samples to arch1_{ARCH_NAME}_samples_gibbs.png")
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
    plt.savefig(f"arch1_{ARCH_NAME}_samples_sa.png")
    plt.savefig(os.path.join(data_dir, f"arch1_{ARCH_NAME}_{hparam_tag}_samples_sa.png"))
    print(f"Saved SA samples to arch1_{ARCH_NAME}_samples_sa.png")
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
    plt.savefig(f"arch1_{ARCH_NAME}_samples_neal.png")
    plt.savefig(os.path.join(data_dir, f"arch1_{ARCH_NAME}_{hparam_tag}_samples_neal.png"))
    print(f"Saved Neal SA samples to arch1_{ARCH_NAME}_samples_neal.png")
    plt.show()

    print(f"\n{ARCH_LABEL} — Complete.")
