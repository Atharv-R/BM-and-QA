'''
FULL-MNIST-test.py: Full 28x28 MNIST on Real D-Wave Zephyr Graph

Tests our Boltzmann Machine architectures (ARCH1/2/3) on the full-size MNIST
dataset (28x28 = 784 visible units) using a Zephyr graph with K matching the
real D-Wave Advantage2 hardware.

D-Wave Advantage2 uses the Zephyr topology Z(K):
  - n(K) = 16K(2K+1) nodes
  - Current Advantage2 prototypes have K≈4..6
  - Z(6) = 1248 nodes → 784 visible + 464 hidden for 28x28 MNIST

The architecture (spectral / tiling / ilp) is selectable as a hyperparameter,
along with all training and architecture-specific hyperparameters.

Data: Full MNIST downloaded via torchvision, binarized at threshold 0.5.
Digit filtering is configurable (default: all digits 0-9).
'''

#%% Imports

import os
import sys
import time
import torch
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import dwave_networkx as dnx
import neal

from bolmaqua import (
    graph_to_bm,
    train_boltzmann_machine_pcd,
    sample_from_bm,
    BM_SimAnn_Sampler,
    device,
    get_zephyr_positions,
    relabel_visible_first,
)

from eval_quality import (
    eval_samples_fullMNIST,
    train_mnist_classifier,
)

from classifierhelpertest import (
    build_native_labelled_graph,
    prepare_classification_batch,
    train_classifier_bm,
    classify_images_fast,
    evaluate_classifier,
    select_and_fuse_label_nodes,
    relabel_fused_graph
)

# Import architecture-specific functions
# (ARCH files now have if __name__ == '__main__' guards, so importing is safe)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module

# ARCH1 functions
_arch1 = import_module('ARCH1-spectral-bisection')
spectral_bisection_partition = _arch1.spectral_bisection_partition
refine_partition_swaps = _arch1.refine_partition_swaps
assign_pixel_order_spatial_arch1 = _arch1.assign_pixel_order_spatial

# ARCH2 functions
_arch2 = import_module('ARCH2-hierarchical-tiling')
tiling_assignment = _arch2.tiling_assignment

# ARCH3 functions
_arch3 = import_module('ARCH3-ilp-bidirectional')
spectral_warm_start = _arch3.spectral_warm_start
ilp_bidirectional_assignment = _arch3.ilp_bidirectional_assignment
greedy_bidirectional_assignment = _arch3.greedy_bidirectional_assignment
assign_pixel_order_spatial_arch3 = _arch3.assign_pixel_order_spatial
HAS_SCIP = _arch3.HAS_SCIP

# Use ARCH1's analyze_architecture (identical across all ARCHs)
analyze_architecture = _arch1.analyze_architecture


# =============================================================================
# Hyperparameters — edit these to configure the experiment
# =============================================================================

#%% 1. Config

# ---- Architecture selection ----
# Options: "spectral" (ARCH1), "tiling" (ARCH2), "ilp" (ARCH3)
architecture = "tiling"

# ---- Zephyr graph parameter ----
# D-Wave Advantage2 prototype: K≈4..6; full Advantage2 target: K≈12+
# Z(6) = 1248 nodes; Z(7) = 1680; Z(8) = 2176
# For 28x28 MNIST (784 visible), need K >= 5 (Z(5)=880, only 96 hidden)
# K=6 recommended: 1248 nodes → 784 visible + 464 hidden
K = 6

# ---- Image configuration ----
grid_shape = (28, 28)  # Full MNIST
num_visible = grid_shape[0] * grid_shape[1]  # 784
binarize_threshold = 0.5

# ---- Digit filtering ----
# Set to None for all digits, or a list like [0, 1] for specific digits
digit_filter = None  # Use all 10 digits

# ---- Classification Mode ----
# Set to True to train discriminative BM for classification instead of generation
use_classification = True  # Set False for generative mode (original behavior)
nodes_per_label = 9  # how many nodes represent each label
classification_loss_weight = 5.0
classification_inference_method = "free_energy"  # QA-compatible class scoring
label_coverage_mode = "hidden_only"  # "hidden_visible", "hidden_only"
label_assignment_time_limit = 120.0  # seconds for native label-node assignment
label_fusion_strategy = 'mst'  # or 'greedy_chain'

# Determine num_classes based on digit_filter
if digit_filter is None:
    num_classes = 10  # All digits 0-9
    label_mapping = None  # No remapping needed
else:
    num_classes = len(digit_filter)
    # Create mapping: e.g., if digit_filter=[3,7], map 3→0, 7→1
    label_mapping = {digit: i for i, digit in enumerate(sorted(digit_filter))}

total_label_nodes = num_classes * nodes_per_label  # ← Total label nodes needed

print(f"Classification: num_classes={num_classes}, nodes_per_label={nodes_per_label}, "
      f"total_label_nodes={total_label_nodes}")
print(f"Classification config: loss_weight={classification_loss_weight}, "
    f"inference_method={classification_inference_method}")
print(f"Label assignment: native nodes, coverage_mode={label_coverage_mode}, "
    f"time_limit={label_assignment_time_limit}s")

# ---- Training hyperparameters ----
lr = 5e-3
weight_decay = 0.00001  # L2 regularization (Adam weight_decay)
batch_size = 64
epochs = 20 
k_steps = 5
persistent_chains = True
eval_every = 3

# ---- Sampling ----
num_samples = 20
gibbs_burn_in = 2000
sa_start_temp = 5.0
sa_end_temp = 0.5
sa_iterations = 64

# ---- QA-like Classification Estimate (Neal) ----
qa_estimate_enabled = False
qa_estimate_num_test_images = 128
qa_estimate_num_reads = 64
qa_estimate_num_sweeps = 100
qa_estimate_beta_range = (0.1, 3.0)
qa_estimate_seed = 42

# ---- Sample Quality Evaluation ----
eval_num_samples = 400       # samples per method for FID evaluation
eval_gibbs_burn_in = 2000    # burn-in for eval Gibbs sampling
eval_sa_start_temp = 5.0     # batched SA start temp for eval
eval_sa_end_temp = 0.1       # batched SA end temp for eval
eval_sa_iterations = 500     # batched SA temperature steps for eval
eval_fid_bootstrap = 100     # bootstrap resamples for FID CI
eval_sampling_methods = ["gibbs", "sa_batched"]  # methods to evaluate

# ---- ARCH1-specific: Spectral Bisection ----
refinement_iters = 1000
vv_penalty = 0.01  # gamma for VV edge penalty in refinement

# ---- ARCH2-specific: Hierarchical Tiling ----
# patch_size must divide both 28 and 28. Valid: 1, 2, 4, 7, 14, 28
patch_size = 7  # 7x7 patches → 4x4=16 tiles; each tile has 49 pixels

# ---- ARCH3-specific: ILP Bidirectional ----
ilp_alpha = 1.0      # weight for min VH-degree of visible nodes
ilp_beta  = 1.0      # weight for min VH-degree of hidden nodes
ilp_gamma = 0.01     # penalty per VV edge
ilp_time_limit = 600  # seconds (larger graph needs more time)


# =============================================================================
# Derived config & validation
# =============================================================================

# Validate K is large enough
n_zephyr = 16 * K * (2 * K + 1)
if n_zephyr < num_visible:
    raise ValueError(
        f"Zephyr Z({K}) has only {n_zephyr} nodes, but need at least "
        f"{num_visible} for {grid_shape[0]}x{grid_shape[1]} images. Increase K."
    )

# Validate architecture choice
VALID_ARCHITECTURES = {"spectral", "tiling", "ilp"}
if architecture not in VALID_ARCHITECTURES:
    raise ValueError(f"Unknown architecture '{architecture}'. Choose from: {VALID_ARCHITECTURES}")

# Validate tiling patch_size for ARCH2
if architecture == "tiling":
    if grid_shape[0] % patch_size != 0 or grid_shape[1] % patch_size != 0:
        raise ValueError(
            f"patch_size={patch_size} does not divide grid_shape={grid_shape}. "
            f"Valid patch sizes for {grid_shape}: "
            f"{[p for p in range(1, grid_shape[0]+1) if grid_shape[0] % p == 0 and grid_shape[1] % p == 0]}"
        )

# Architecture names and labels
ARCH_NAMES = {
    "spectral": ("spectral", "ARCH1: Spectral Bisection"),
    "tiling":   ("tiling",   "ARCH2: Hierarchical Tiling"),
    "ilp":      ("ilp",      "ARCH3: ILP Bidirectional"),
}
ARCH_NAME, ARCH_LABEL = ARCH_NAMES[architecture]

# Build hyperparam tag for filenames
train_method = "PCD" if persistent_chains else "CD"

# Base tag (common hyperparams)
hparam_base = f"{train_method}_K{K}_lr{lr}_l2{weight_decay}_bs{batch_size}_ep{epochs}_k{k_steps}"

# Architecture-specific suffix
if architecture == "spectral":
    hparam_arch = f"_refine{refinement_iters}_gamma{vv_penalty}"
elif architecture == "tiling":
    hparam_arch = f"_patch{patch_size}"
elif architecture == "ilp":
    hparam_arch = f"_a{ilp_alpha}_b{ilp_beta}_g{ilp_gamma}"

hparam_tag = hparam_base + hparam_arch

# Digit filter tag for filenames
if digit_filter is not None:
    digits_str = "".join(str(d) for d in sorted(digit_filter))
    digit_tag = f"_digits{digits_str}"
else:
    digits_str = "all"
    digit_tag = "_digitsAll"

# File prefix
file_prefix = f"fullmnist_classifier_{ARCH_NAME}"

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

print("=" * 70)
print(f"  FULL MNIST TEST — {ARCH_LABEL}")
print(f"  Zephyr Z({K}): {n_zephyr} nodes for {grid_shape[0]}x{grid_shape[1]} images")
print("=" * 70)
print(f"  Architecture: {architecture}")
print(f"  Grid shape: {grid_shape} ({num_visible} visible units)")
print(f"  Digits: {digits_str}")
print(f"  Training: lr={lr}, l2={weight_decay}, epochs={epochs}, k={k_steps}, bs={batch_size}")
print(f"  Method: {train_method}")
print(f"  QA estimate: enabled={qa_estimate_enabled}, eval_images={qa_estimate_num_test_images}, "
    f"reads={qa_estimate_num_reads}, sweeps={qa_estimate_num_sweeps}")
if architecture == "spectral":
    print(f"  ARCH1 params: refinement_iters={refinement_iters}, vv_penalty={vv_penalty}")
elif architecture == "tiling":
    print(f"  ARCH2 params: patch_size={patch_size}")
elif architecture == "ilp":
    print(f"  ARCH3 params: alpha={ilp_alpha}, beta={ilp_beta}, gamma={ilp_gamma}, time_limit={ilp_time_limit}s")
    print(f"  SCIP available: {HAS_SCIP}")
print(f"  Hparam tag: {hparam_tag}")
print("=" * 70)


# =============================================================================
# 2. Data Loading — Full MNIST from torchvision
# =============================================================================

print("\nLoading full MNIST dataset from torchvision...")

# Download MNIST (will cache in ./data/MNIST/)
train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transforms.ToTensor())
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

def prepare_mnist(dataset, digit_filter=None, threshold=0.5, label_mapping=None):
    """Extract, optionally filter, and binarize MNIST data. Returns images AND labels."""
    images = dataset.data.float() / 255.0
    labels = dataset.targets

    # Filter digits
    if digit_filter is not None:
        mask = torch.zeros(len(labels), dtype=torch.bool)
        for d in digit_filter:
            mask |= (labels == d)
        images = images[mask]
        labels = labels[mask]
        
        # Remap labels to consecutive integers [0, num_classes-1]
        if label_mapping is not None:
            labels = torch.tensor([label_mapping[l.item()] for l in labels], dtype=torch.long)

    # Binarize
    images = (images >= threshold).float()
    images = images.view(images.size(0), -1)

    return images, labels

# Update the calls:
train_data, train_labels = prepare_mnist(train_dataset, digit_filter, binarize_threshold, label_mapping)
test_data, test_labels   = prepare_mnist(test_dataset,  digit_filter, binarize_threshold, label_mapping)

print(f"  Train: {train_data.shape[0]} samples, {train_data.shape[1]} features")
print(f"  Test:  {test_data.shape[0]} samples, {test_data.shape[1]} features")
print(f"  Digits: {digits_str}")
print(f"  Mode: CLASSIFICATION (num_classes={num_classes})")


# Create datasets and loaders

train_dataset_final = TensorDataset(train_data, train_labels)
test_dataset_final = TensorDataset(test_data, test_labels)


loader = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset_final, batch_size=batch_size, shuffle=False)


# =============================================================================
# 3. Architecture Construction
# =============================================================================

print(f"\nGenerating Zephyr graph Z({K})...")
t0 = time.time()
G_zephyr = dnx.zephyr_graph(K)
n_total = G_zephyr.number_of_nodes()
n_edges = G_zephyr.number_of_edges()
degrees = dict(G_zephyr.degree())
print(f"  Nodes: {n_total}, Edges: {n_edges}")
print(f"  Degree: min={min(degrees.values())}, max={max(degrees.values())}, "
      f"mean={np.mean(list(degrees.values())):.1f}")
print(f"  Visible units needed: {num_visible}, Hidden units available: {n_total - num_visible}")

if architecture == "spectral":
    # ---- ARCH1: Spectral Bisection + Refinement ----
    print(f"\n--- ARCH1: Spectral Bisection ---")
    print("Step 1: Spectral bisection partition...")
    visible_set = spectral_bisection_partition(G_zephyr, num_visible)

    print("Step 2: Local swap refinement...")
    visible_set = refine_partition_swaps(
        G_zephyr, visible_set, num_visible,
        max_iters=refinement_iters, gamma=vv_penalty
    )

    print("Step 3: Spatial pixel assignment...")
    visible_in_pixel_order = assign_pixel_order_spatial_arch1(G_zephyr, visible_set, grid_shape)
    hidden_nodes = [n for n in sorted(G_zephyr.nodes()) if n not in visible_set]

elif architecture == "tiling":
    # ---- ARCH2: Hierarchical Tiling ----
    print(f"\n--- ARCH2: Hierarchical Tiling (patch={patch_size}x{patch_size}) ---")
    print("Steps 1-4: Tiling assignment...")
    visible_in_pixel_order, hidden_nodes = tiling_assignment(G_zephyr, grid_shape, patch_size)

elif architecture == "ilp":
    # ---- ARCH3: ILP Bidirectional ----
    print(f"\n--- ARCH3: ILP Bidirectional ---")
    print("Step 1: Computing spectral warm start...")
    warm_start = spectral_warm_start(G_zephyr, num_visible)
    ws_vh = sum(1 for u, v in G_zephyr.edges() if (u in warm_start) != (v in warm_start))
    print(f"  Warm start VH edges: {ws_vh}")

    if HAS_SCIP:
        print("Step 2: Solving ILP with SCIP...")
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
        print("Step 2: Greedy bidirectional optimization (SCIP not available)...")
        visible_set = greedy_bidirectional_assignment(
            G_zephyr, num_visible, gamma=ilp_gamma)

    print("Step 3: Spatial pixel assignment...")
    visible_in_pixel_order = assign_pixel_order_spatial_arch3(G_zephyr, visible_set, grid_shape)
    hidden_nodes = [n for n in sorted(G_zephyr.nodes()) if n not in visible_set]

num_hidden = len(hidden_nodes)
print(f"\nRelabeling graph (visible=0..{num_visible-1}, hidden={num_visible}..{num_visible+num_hidden-1})...")
G_relabeled, mapping = relabel_visible_first(G_zephyr, visible_in_pixel_order)

# Build node labels
node_labels = {}
for i in range(num_visible):
    node_labels[i] = 'visible'
for i in range(num_visible, num_visible + num_hidden):
    node_labels[i] = 'hidden'

t_arch = time.time() - t0
print(f"Architecture construction time: {t_arch:.1f}s")


# =============================================================================
# 4. Architecture Analysis
# =============================================================================

visible_relabeled = list(range(num_visible))
hidden_relabeled = list(range(num_visible, num_visible + num_hidden))

# Analyze the base pixel/hidden partition before native label reassignment.
stats = analyze_architecture(
    G_relabeled,
    visible_relabeled,
    hidden_relabeled,
    f"{ARCH_LABEL} (base partition)"
)

# =============================================================================
# 4a. Add Label Nodes for Classification via FUSION
# =============================================================================

print(f"\n--- Selecting and Fusing Label Nodes ---")

# Step 1: Select and fuse
G_fused, label_node_groups_original, fused_label_ids, fusion_stats = (
    select_and_fuse_label_nodes(
        G_relabeled,
        visible_relabeled,
        hidden_relabeled,
        num_classes=num_classes,
        nodes_per_label=nodes_per_label,
        coverage_mode=label_coverage_mode,
        fusion_strategy='mst',  # or 'greedy_chain'
        verbose=True,
    )
)

# Step 2: Relabel for training
G_relabeled, label_node_groups, remaining_hidden, node_labels, fusion_mapping = (
    relabel_fused_graph(
        G_fused,
        visible_relabeled,
        fused_label_ids,
        label_node_groups_original,
    )
)

num_visible_total = num_visible + num_classes  # pixels + labels (now super-nodes)
num_hidden = len(remaining_hidden)

print(f"\n  Final architecture:")
print(f"    Pixels: {num_visible}")
print(f"    Labels: {num_classes} (fused from {num_classes * nodes_per_label} original nodes)")
print(f"    Hidden: {num_hidden}")
print(f"    Total visible (pixels + labels): {num_visible_total}")


# ANALYZE NODE FUSION RESULTS
print(f"\n{'='*70}")
print("  FUSED LABEL NODE ANALYSIS")
print(f"{'='*70}")
for class_idx, super_node_id in enumerate(range(num_visible, num_visible + num_classes)):
    degree = G_relabeled.degree(super_node_id)
    pixel_connections = sum(1 for n in G_relabeled.neighbors(super_node_id) 
                           if n < num_visible)
    hidden_connections = sum(1 for n in G_relabeled.neighbors(super_node_id) 
                            if n >= num_visible + num_classes)
    
    print(f"  Class {class_idx} (node {super_node_id}):")
    print(f"    Total degree: {degree}")
    print(f"    Pixel connections: {pixel_connections}")
    print(f"    Hidden connections: {hidden_connections}")
print(f"{'='*70}\n")

# =============================================================================
# 5. Model Initialization
# =============================================================================

print("Initializing CustomBoltzmannMachine...")
model = graph_to_bm(G_relabeled, node_labels)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model on {device}, total parameters: {total_params:,}")
print(f"  Visible: {num_visible_total}, Hidden: {num_hidden}")


# =============================================================================
# 6. Training
# =============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
print(f"\nStarting CLASSIFICATION Training (lr={lr}, epochs={epochs}, k={k_steps})...")

training_history = train_classifier_bm(
    model, loader, optimizer,
    num_epochs=epochs, k_steps=k_steps,
    label_node_groups=label_node_groups,
    batch_size=batch_size, step_size=lr, 
    num_classes=num_classes,
    nodes_per_label=1,
    classification_loss_weight=classification_loss_weight,
)

# =============================================================================
# 6c. Classification Evaluation (if in classification mode)
# =============================================================================

aggregation_method = 'average'  # 'average' or 'majority'

print(f"\n{'='*70}")
print("  CLASSIFICATION EVALUATION")
print(f"{'='*70}")

model.eval()
with torch.inference_mode():
    train_acc, train_per_class = evaluate_classifier(
        model, 
        DataLoader(train_dataset_final, batch_size=batch_size, shuffle=False),
        label_node_groups,
        num_gibbs_steps=10,
        num_classes=num_classes,
        nodes_per_label=nodes_per_label, 
        aggregation=aggregation_method,
        inference_method=classification_inference_method,
    )

    test_acc, test_per_class = evaluate_classifier(
        model, test_loader, 
        label_node_groups, 
        num_gibbs_steps=10,
        num_classes=num_classes,
        nodes_per_label=nodes_per_label,
        aggregation=aggregation_method,
        inference_method=classification_inference_method,
    )

#%% 9. Visualize Some Predictions

print("\nVisualizing predictions on test images...")
sample_batch = next(iter(test_loader))
sample_pixels, sample_labels = sample_batch[0][:9].to(device), sample_batch[1][:9]

with torch.inference_mode():
    preds, confs = classify_images_fast(
        model, sample_pixels, 
        label_node_groups,
        num_gibbs_steps=50,
        num_classes=num_classes,
        nodes_per_label=nodes_per_label,
        aggregation=aggregation_method,
        inference_method=classification_inference_method,
    )

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

fig.suptitle(f"FULL MNIST Classifier {ARCH_LABEL} — Test Predictions (Acc={100*test_acc:.1f}%)")
plt.tight_layout()
plt.savefig(f"FULL MNIST Classifier {ARCH_NAME}_predictions.png")
plt.savefig(os.path.join(data_dir, f"FULL MNIST Classifier {ARCH_NAME}_{hparam_tag}_predictions.png"))
print(f"Saved predictions to FULL MNIST Classifier {ARCH_NAME}_predictions.png")
plt.show()

print(f"\n{'='*60}")
print(f"CLASSIFICATION RESULTS")
print(f"{'='*60}")
print(f"Train Accuracy: {100*train_acc:.2f}%")
print(f"Test Accuracy:  {100*test_acc:.2f}%")
print(f"Per-class test accuracy:")
for digit, acc in test_per_class.items():
    print(f"  Digit {digit}: {100*acc:.2f}%")
print(f"{'='*60}\n")


def estimate_classifier_with_neal_conditional_sampling(
    model,
    pixels,
    labels,
    num_pixel_nodes,
    num_classes,
    nodes_per_label,
    num_reads=64,
    num_sweeps=100,
    beta_range=(0.1, 3.0),
    seed=None,
):
    """
    Estimate QA-deployable classification by clamping image pixels and sampling
    free label + hidden variables with Neal SA, then decoding the sampled labels.
    """
    import dimod
     

    model.eval()

    with torch.no_grad():
        W_vv, W_hh, W_vh = model._get_masked_weights()
        W_vv = W_vv.cpu().numpy()
        W_hh = W_hh.cpu().numpy()
        W_vh = W_vh.cpu().numpy()
        b_v = model.b_v.cpu().numpy()
        b_h = model.b_h.cpu().numpy()

    nv = model.num_visible
    nh = model.num_hidden

    linear = {}
    quadratic = {}

    for i in range(nv):
        linear[i] = -float(b_v[i])
    for j in range(nh):
        linear[nv + j] = -float(b_h[j])

    for i in range(nv):
        for j in range(i + 1, nv):
            w = float(W_vv[i, j])
            if w != 0.0:
                quadratic[(i, j)] = -w

    for i in range(nh):
        for j in range(i + 1, nh):
            w = float(W_hh[i, j])
            if w != 0.0:
                quadratic[(nv + i, nv + j)] = -w

    for i in range(nv):
        for j in range(nh):
            w = float(W_vh[i, j])
            if w != 0.0:
                quadratic[(i, nv + j)] = -w

    base_bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)
    sampler = neal.SimulatedAnnealingSampler()

    all_preds = []
    all_labels = []
    all_confidences = []
    actual_label_nodes = num_classes  # Not num_classes * nodes_per_label!

    print(f"\n{'='*70}")
    print("  QA-LIKE CONDITIONAL SAMPLING ESTIMATE (NEAL)")
    print(f"{'='*70}")
    print(f"Evaluating {len(labels)} test images with clamped-pixel Neal sampling...")

    for idx in range(len(labels)):
        pixel_vec = pixels[idx].cpu().numpy().astype(np.int8)
        bqm = base_bqm.copy()
        
        # Clamp pixels
        for pixel_idx in range(num_pixel_nodes):
            bqm.fix_variable(pixel_idx, int(pixel_vec[pixel_idx]))
        
        # Sample
        sampleset = sampler.sample(
            bqm, num_reads=num_reads, num_sweeps=num_sweeps,
            beta_range=beta_range, seed=seed,
        )
        var_list = list(sampleset.variables)
        
        # Extract label samples (now just num_classes nodes, not num_classes * nodes_per_label)
        label_samples = np.zeros((num_reads, actual_label_nodes), dtype=np.float32)
        for class_idx in range(num_classes):
            visible_label_idx = num_pixel_nodes + class_idx  # Single node per class
            if visible_label_idx in var_list:
                sample_col = var_list.index(visible_label_idx)
                label_samples[:, class_idx] = sampleset.record.sample[:, sample_col]
        
        # Score: just use the label node activations directly
        class_scores = label_samples.mean(axis=0)  # Average across reads
        
        pred_class = int(np.argmax(class_scores))
        confidence = float(class_scores[pred_class])
        all_preds.append(pred_class)
        all_labels.append(int(labels[idx].item()))
        all_confidences.append(confidence)

        if (idx + 1) % max(1, len(labels) // 10) == 0:
            print(f"  Processed {idx+1}/{len(labels)} QA-like samples...")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = float(np.mean(all_preds == all_labels))

    per_class_acc = {}
    for class_idx in range(num_classes):
        class_mask = all_labels == class_idx
        if class_mask.any():
            per_class_acc[class_idx] = float(np.mean(all_preds[class_mask] == all_labels[class_mask]))
        else:
            per_class_acc[class_idx] = 0.0

    print(f"\n  QA-like Neal Accuracy: {100*accuracy:.2f}%")
    print(f"  Mean QA-like confidence: {100*np.mean(all_confidences):.2f}%")
    for class_idx in range(num_classes):
        print(f"  Class {class_idx} QA-like Accuracy: {100*per_class_acc[class_idx]:.2f}%")

    return {
        'accuracy': accuracy,
        'per_class': per_class_acc,
        'mean_confidence': float(np.mean(all_confidences)),
        'num_eval_images': int(len(labels)),
        'num_reads': int(num_reads),
        'num_sweeps': int(num_sweeps),
        'beta_range': beta_range,
    }


qa_like_results = None
if qa_estimate_enabled:
    qa_eval_count = min(qa_estimate_num_test_images, len(test_data))
    qa_indices = torch.randperm(len(test_data))[:qa_eval_count]
    qa_test_pixels = test_data[qa_indices]
    qa_test_labels = test_labels[qa_indices]
    qa_like_results = estimate_classifier_with_neal_conditional_sampling(
        model=model,
        pixels=qa_test_pixels,
        labels=qa_test_labels,
        num_pixel_nodes=num_visible,
        num_classes=num_classes,
        nodes_per_label=nodes_per_label,
        num_reads=qa_estimate_num_reads,
        num_sweeps=qa_estimate_num_sweeps,
        beta_range=qa_estimate_beta_range,
        seed=qa_estimate_seed,
    )

# Save classification results
classification_results = {
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'train_per_class': train_per_class,
    'test_per_class': test_per_class,
    'qa_like_neal_estimate': qa_like_results,
}

# =============================================================================
# 6a. Save model checkpoint (will be updated with quality metrics later)
# =============================================================================

model_path = os.path.join(data_dir, f"{file_prefix}_{hparam_tag}{digit_tag}_model.pt")

# Build hyperparams dict with all settings
hyperparams = {
    'architecture': architecture,
    'K': K, 'grid_shape': grid_shape,
    'lr': lr, 'weight_decay': weight_decay, 'l2_reg': weight_decay,
    'batch_size': batch_size, 'epochs': epochs, 'k_steps': k_steps,
    'persistent_chains': persistent_chains,
    'binarize_threshold': binarize_threshold,
    'digit_filter': digit_filter,
    'num_visible': num_visible, 'num_hidden': num_hidden,
    'num_train_samples': len(train_data),
    'num_test_samples': len(test_data),
}

# Add architecture-specific params
if architecture == "spectral":
    hyperparams.update({
        'refinement_iters': refinement_iters,
        'vv_penalty': vv_penalty,
    })
elif architecture == "tiling":
    hyperparams.update({
        'patch_size': patch_size,
    })
elif architecture == "ilp":
    hyperparams.update({
        'ilp_alpha': ilp_alpha,
        'ilp_beta': ilp_beta,
        'ilp_gamma': ilp_gamma,
        'ilp_time_limit': ilp_time_limit,
    })

# Add to hyperparams dict (after the existing entries):
hyperparams.update({
    'use_classification': use_classification,
    'num_classes': num_classes if use_classification else None,
    'label_nodes': label_node_groups if use_classification else None,
    'label_assignment_mode': 'native_coverage' if use_classification else None,
    'label_coverage_mode': label_coverage_mode if use_classification else None,
    'label_assignment_time_limit': label_assignment_time_limit if use_classification else None,
    'classification_loss_weight': classification_loss_weight if use_classification else None,
    'classification_inference_method': classification_inference_method if use_classification else None,
    'fusion_stats': fusion_stats,
    'qa_estimate_enabled': qa_estimate_enabled if use_classification else None,
    'qa_estimate_num_test_images': qa_estimate_num_test_images if use_classification else None,
    'qa_estimate_num_reads': qa_estimate_num_reads if use_classification else None,
    'qa_estimate_num_sweeps': qa_estimate_num_sweeps if use_classification else None,
    'qa_estimate_beta_range': qa_estimate_beta_range if use_classification else None,
})

# Early save (will be overwritten after quality evaluation with full metrics)
torch.save({
    'model_state_dict': model.state_dict(),
    'training_history': training_history,
    'arch_name': ARCH_NAME,
    'arch_label': ARCH_LABEL,
    'hyperparams': hyperparams,
    'graph_edges': list(G_relabeled.edges()),
    'node_labels': node_labels,
    'hparam_tag': hparam_tag,
    'classification_results': classification_results,
}, model_path)
print(f"Saved initial model checkpoint to {model_path}")


