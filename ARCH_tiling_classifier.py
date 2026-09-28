'''
ARCH2: Hierarchical Tiling with Dedicated Hidden Clusters

Strategy: Partition the Zephyr graph into spatial "tiles" that correspond
to image patches. Within each tile, assign some nodes as visible (for that
patch's pixels) and the rest as hidden. This creates a locally-connected
architecture where each hidden cluster serves a specific image region.

Rationale: Convolutional/locally-connected architectures are known to learn
image distributions well. By partitioning the Zephyr graph into tiles that
map to image patches, each tile's hidden units act as local feature detectors.
Inter-tile HH edges provide long-range correlation capacity that pure CNNs lack.
The key difference from Strategy 1 (global max-cut) is that visible nodes are
distributed throughout the graph, each guaranteed nearby hidden neighbors.
'''

#%% defs

import os
import time
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
import dwave_networkx as dnx

from bolmaqua import (
    graph_to_bm,
    train_boltzmann_machine_pcd,
    sample_from_bm,
    BM_SimAnn_Sampler,
    GRID_SHAPE,
    device,
    get_zephyr_positions,
    relabel_visible_first,
)

from optihelper import visualize_node_assignment_on_zephyr

from ARCH_classifier import (
    add_label_nodes_to_graph,
    prepare_classification_batch,
    train_classifier_bm,
    classify_images_fast,
    evaluate_classifier,
)

ARCH_NAME = "tiling"
ARCH_LABEL = "ARCH2: Hierarchical Tiling"


def load_data_with_labels(grid_shape=GRID_SHAPE):
    """
    Load MNIST data (digits 0 and 1) with labels.
    
    Returns:
        train_feats: (N, num_pixels) tensor
        train_labels: (N,) tensor with values 0 or 1
    """
    try:
        train_feats = np.load(f'mnist{grid_shape[0]}x{grid_shape[1]}_trainfeats.npy')
        train_labels = np.load(f'mnist{grid_shape[0]}x{grid_shape[1]}_trainlabels.npy')
    except FileNotFoundError:
        print(f"Error: Could not find mnist{grid_shape[0]}x{grid_shape[1]} data files.")
        return None, None
    
    # Filter for digits 0 and 1
    mask = (train_labels == 0) | (train_labels == 1)
    train_feats = train_feats[mask]
    train_labels = train_labels[mask]
    
    # Binarize pixels
    train_feats = (train_feats > 0.5).astype(np.float32)
    
    return torch.from_numpy(train_feats), torch.from_numpy(train_labels).long()


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



def tiling_assignment(G, grid_shape, patch_size):
    """
    Partition the Zephyr graph into spatial tiles and assign visible/hidden
    within each tile. Each tile corresponds to one image patch.

    Args:
        G: Zephyr NetworkX graph
        grid_shape: (nrows, ncols) of the target image
        patch_size: side length of each square patch (e.g. 3 for 3x3 patches)

    Returns:
        visible_in_pixel_order: list of node IDs, one per pixel, in row-major order
        hidden_nodes: list of remaining node IDs
    """
    nrows, ncols = grid_shape
    num_visible = nrows * ncols
    assert nrows % patch_size == 0 and ncols % patch_size == 0, \
        f"grid_shape {grid_shape} must be divisible by patch_size {patch_size}"

    ntiles_r = nrows // patch_size
    ntiles_c = ncols // patch_size
    num_tiles = ntiles_r * ntiles_c
    pix_per_tile = patch_size * patch_size

    print(f"  Tiling: {ntiles_r}x{ntiles_c} = {num_tiles} tiles, "
          f"{pix_per_tile} pixels/tile, patch_size={patch_size}")

    pos = get_zephyr_positions(G)
    nodes = sorted(G.nodes())
    all_coords = np.array([pos[n] for n in nodes])
    xmin, xmax = all_coords[:, 0].min(), all_coords[:, 0].max()
    ymin, ymax = all_coords[:, 1].min(), all_coords[:, 1].max()

    # ---- Step 1: Compute tile centers in layout space ----
    tile_centers = np.zeros((num_tiles, 2))
    for tr in range(ntiles_r):
        for tc in range(ntiles_c):
            tid = tr * ntiles_c + tc
            # Center of this tile's pixel group, mapped to layout space
            pixel_r_center = (tr + 0.5) * patch_size
            pixel_c_center = (tc + 0.5) * patch_size
            lx = xmin + pixel_c_center / ncols * (xmax - xmin)
            ly = ymax - pixel_r_center / nrows * (ymax - ymin)
            tile_centers[tid] = [lx, ly]

    # ---- Step 2: Assign each node to nearest tile center ----
    tile_nodes = defaultdict(list)
    for node in nodes:
        coord = np.array(pos[node])
        dists = np.sum((tile_centers - coord) ** 2, axis=1)
        nearest = np.argmin(dists)
        tile_nodes[nearest].append(node)

    tile_sizes = [len(tile_nodes[t]) for t in range(num_tiles)]
    print(f"  Tile sizes before rebalancing: min={min(tile_sizes)} "
          f"mean={np.mean(tile_sizes):.1f} max={max(tile_sizes)}")

    # ---- Step 3: Rebalance — ensure each tile has >= pix_per_tile nodes ----
    for t in range(num_tiles):
        attempts = 0
        while len(tile_nodes[t]) < pix_per_tile and attempts < 100:
            attempts += 1
            # Find nearest tile with excess
            best_donor = None
            best_dist = float('inf')
            for t2 in range(num_tiles):
                if t2 == t or len(tile_nodes[t2]) <= pix_per_tile:
                    continue
                d = np.sum((tile_centers[t] - tile_centers[t2]) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_donor = t2
            if best_donor is None:
                break
            # Move the closest node from donor to needy tile
            donor_nodes = tile_nodes[best_donor]
            closest = min(donor_nodes,
                          key=lambda n: np.sum((np.array(pos[n]) - tile_centers[t]) ** 2))
            donor_nodes.remove(closest)
            tile_nodes[t].append(closest)

    tile_sizes = [len(tile_nodes[t]) for t in range(num_tiles)]
    print(f"  Tile sizes after rebalancing:  min={min(tile_sizes)} "
          f"mean={np.mean(tile_sizes):.1f} max={max(tile_sizes)}")

    underfilled = sum(1 for s in tile_sizes if s < pix_per_tile)
    if underfilled:
        print(f"  ⚠️  {underfilled} tiles still have fewer than {pix_per_tile} nodes")

    # ---- Step 4: Within each tile, select visible nodes ----
    # Priority: nodes with highest degree in the tile's LOCAL subgraph
    # (maximizes within-tile VH connectivity)
    pixel_node_map = {}   # global_pixel_idx -> node
    all_visible = set()

    for tid in range(num_tiles):
        tr = tid // ntiles_c
        tc = tid % ntiles_c
        tnodes = tile_nodes[tid]

        # Sort by degree within the tile's local subgraph
        tile_subgraph = G.subgraph(tnodes)
        tnodes_by_internal_deg = sorted(
            tnodes,
            key=lambda n: tile_subgraph.degree(n),
            reverse=True
        )
        vis_candidates = tnodes_by_internal_deg[:pix_per_tile]

        # Build pixel positions for this tile (in layout space)
        pixel_positions = []
        for pr in range(patch_size):
            for pc in range(patch_size):
                gr = tr * patch_size + pr
                gc = tc * patch_size + pc
                pix_idx = gr * ncols + gc
                lx = xmin + (gc + 0.5) / ncols * (xmax - xmin)
                ly = ymax - (gr + 0.5) / nrows * (ymax - ymin)
                pixel_positions.append((pix_idx, lx, ly))

        # Greedy spatial matching: each pixel → nearest unassigned visible candidate
        used = set()
        for pix_idx, lx, ly in pixel_positions:
            target = np.array([lx, ly])
            best_node = None
            best_dist = float('inf')
            for node in vis_candidates:
                if node in used:
                    continue
                d = np.sum((np.array(pos[node]) - target) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_node = node
            if best_node is not None:
                pixel_node_map[pix_idx] = best_node
                all_visible.add(best_node)
                used.add(best_node)

    # ---- Step 5: Handle any missing pixels (fallback for under-filled tiles) ----
    missing = [i for i in range(num_visible) if i not in pixel_node_map]
    if missing:
        print(f"  Fallback: assigning {len(missing)} missing pixels from global pool")
        remaining = [n for n in nodes if n not in all_visible]
        for pix_idx in missing:
            gr, gc = pix_idx // ncols, pix_idx % ncols
            lx = xmin + (gc + 0.5) / ncols * (xmax - xmin)
            ly = ymax - (gr + 0.5) / nrows * (ymax - ymin)
            target = np.array([lx, ly])
            best_node = min(remaining,
                            key=lambda n: np.sum((np.array(pos[n]) - target) ** 2))
            pixel_node_map[pix_idx] = best_node
            all_visible.add(best_node)
            remaining.remove(best_node)

    visible_in_pixel_order = [pixel_node_map[i] for i in range(num_visible)]
    hidden_nodes = [n for n in nodes if n not in all_visible]

    # ---- Report tile-level VH connectivity ----
    tile_vh = []
    for tid in range(num_tiles):
        tnodes = tile_nodes[tid]
        vis_in_tile = [n for n in tnodes if n in all_visible]
        hid_in_tile = [n for n in tnodes if n not in all_visible]
        local_vh = 0
        vis_set_tile = set(vis_in_tile)
        for v in vis_in_tile:
            for nb in G.neighbors(v):
                if nb in hid_in_tile:
                    local_vh += 1
        tile_vh.append(local_vh)
    print(f"  Intra-tile VH edges: min={min(tile_vh)} mean={np.mean(tile_vh):.1f} max={max(tile_vh)}")

    return visible_in_pixel_order, hidden_nodes



if __name__ == '__main__':
    print("=" * 60)
    print(ARCH_LABEL)
    print("=" * 60)

    #%% 1. Config

    K = 3  # Zephyr graph parameter
    grid_shape = GRID_SHAPE  # (12, 12) from bolmaqua
    num_visible = grid_shape[0] * grid_shape[1]
    num_classes = 2

    # Training hyperparameters (matching test-RBM-as-custom-BM.py)
    lr = 5e-3
    weight_decay = 0.00001
    batch_size = 64
    epochs = 2
    k_steps = 15
    persistent_chains = True

    # Sampling
    num_samples = 9
    gibbs_burn_in = 2000
    sa_start_temp = 10.0
    sa_end_temp = 0.1
    sa_iterations = 16

    # Strategy-specific
    patch_size = 3  # 3x3 patches → 4x4=16 tiles for 12x12 image; each tile has 9 pixels

    # Classification inference
    inference_gibbs_steps = 5

    # Build hyperparam tag for filenames
    train_method = "PCD" if persistent_chains else "CD"
    hparam_tag = f"{train_method}_K{K}_lr{lr}_l2{weight_decay}_bs{batch_size}_ep{epochs}_k{k_steps}_patch{patch_size}"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Zephyr K={K}, grid_shape={grid_shape}, num_visible={num_visible}")
    print(f"Patch size: {patch_size}x{patch_size} → "
          f"{grid_shape[0]//patch_size}x{grid_shape[1]//patch_size} tiles")
    print(f"Training: lr={lr}, epochs={epochs}, k_steps={k_steps}, batch_size={batch_size}")
    print(f"Hparam tag: {hparam_tag}")

    #%% 2. Data Loading

    print("\nLoading MNIST data...")
    train_pixels, train_labels = load_data_with_labels(grid_shape)
    if train_pixels is None:
        raise RuntimeError("Failed to load data")
    
    # Split into train/test
    n_total = len(train_pixels)
    n_train = int(0.8 * n_total)

    train_pixels_split = train_pixels[:n_train]
    train_labels_split = train_labels[:n_train]
    test_pixels = train_pixels[n_train:]
    test_labels = train_labels[n_train:]

    train_dataset = TensorDataset(train_pixels_split, train_labels_split)
    test_dataset = TensorDataset(test_pixels, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_pixels_split)} | Test: {len(test_pixels)}")
    print(f"Class distribution - Train: 0={torch.sum(train_labels_split==0).item()}, "
          f"1={torch.sum(train_labels_split==1).item()}")


    #%% 3. Architecture Construction

    print(f"\nGenerating Zephyr graph Z({K})...")
    t0 = time.time()
    G_zephyr = dnx.zephyr_graph(K)
    n_total = G_zephyr.number_of_nodes()
    n_edges = G_zephyr.number_of_edges()
    print(f"  Nodes: {n_total}, Edges: {n_edges}")
    print(f"  Degree: min={min(dict(G_zephyr.degree()).values())}, "
          f"max={max(dict(G_zephyr.degree()).values())}")

    print("\nStep 1-4: Tiling assignment...")
    visible_in_pixel_order, hidden_nodes = tiling_assignment(G_zephyr, grid_shape, patch_size)
    num_hidden = len(hidden_nodes)

    print(f"\nStep 5: Relabeling graph (visible=0..{num_visible-1}, hidden={num_visible}..{num_visible+num_hidden-1})...")
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
    
     # Check coverage
    all_assigned = set(visible_relabeled) | set(hidden_relabeled)
    all_graph = set(G_relabeled.nodes())
    missing = all_graph - all_assigned
    extra = all_assigned - all_graph

    if missing:
        print(f"  ⚠️ {len(missing)} graph nodes not assigned to visible/hidden: {sorted(missing)[:10]}...")
    if extra:
        print(f"  ⚠️ {len(extra)} assigned nodes not in graph: {sorted(extra)[:10]}...")
    if not missing and not extra:
        print(f"  ✓ Perfect coverage: all graph nodes are assigned")

    print("=" * 60)
    
    G_extended, label_nodes, node_labels_extended = add_label_nodes_to_graph(
        G_relabeled, visible_relabeled, hidden_relabeled, num_classes=num_classes
    )

    #%% 5. Model Initialization

    print("Initializing CustomBoltzmannMachine...")
    model = graph_to_bm(G_extended, node_labels_extended)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model on {device}, total parameters: {total_params:,}")

    #%% 6. Training (PCD)

    print(f"\nStarting PCD Training (lr={lr}, epochs={epochs}, k={k_steps})...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    training_history = train_classifier_bm(
        model, train_loader, optimizer, 
        num_epochs=epochs, k_steps=k_steps,
        label_nodes=label_nodes,
        batch_size=batch_size, step_size=lr
    )

    train_acc, train_per_class = evaluate_classifier(
        model, train_loader, label_nodes, num_gibbs_steps=inference_gibbs_steps
    )
    
    test_acc, test_per_class = evaluate_classifier(
        model, test_loader, label_nodes, num_gibbs_steps=inference_gibbs_steps
    )


    #%% 8. Save Model
    
    model_path = os.path.join(data_dir, f"{ARCH_NAME}_classifier_{hparam_tag}_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'test_accuracy': test_acc,
        'test_per_class': test_per_class,
        'label_nodes': label_nodes,
        'hyperparams': {
            'K': K, 'num_pixels': num_visible, 'num_hidden': num_hidden,
            'num_classes': num_classes, 'lr': lr, 'epochs': epochs, 'k_steps': k_steps,
        },
        'graph_edges': list(G_extended.edges()),
        'node_labels': node_labels_extended,
    }, model_path)
    print(f"Saved model to {model_path}")


    #%% 9. Visualize Some Predictions
    print("\nVisualizing predictions on test images...")
    sample_batch = next(iter(test_loader))
    sample_pixels, sample_labels = sample_batch[0][:9].to(device), sample_batch[1][:9]

    # Use FAST inference
    preds, confs = classify_images_fast(model, sample_pixels, label_nodes, 
                                        num_gibbs_steps=50)  # Lower since it's mean-field

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        img = sample_pixels[i].cpu().numpy().reshape(grid_shape)
        ax.imshow(img, cmap='gray')
        true_label = sample_labels[i].item()
        pred_label = preds[i].item()
        conf = confs[i].item()
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}, Pred: {pred_label}\nConf: {conf:.2f}", color=color)
        ax.axis('off')

    fig.suptitle(f"{ARCH_LABEL} — Test Predictions (Acc={100*test_acc:.1f}%)")
    plt.tight_layout()
    plt.savefig(f"{ARCH_NAME}_classifier_predictions.png")
    plt.savefig(os.path.join(data_dir, f"{ARCH_NAME}_{hparam_tag}_predictions.png"))
    print(f"Saved predictions to {ARCH_NAME}_predictions.png")
    plt.show()

    print(f"\n{ARCH_LABEL} — Complete.")
