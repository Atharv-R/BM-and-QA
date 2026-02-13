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

from bolmaqua import (
    graph_to_bm,
    train_boltzmann_machine_pcd,
    sample_from_bm,
    BM_SimAnn_Sampler,
    device,
    get_zephyr_positions,
    relabel_visible_first,
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

# ---- Training hyperparameters ----
lr = 1e-4
weight_decay = 0.0001  # L2 regularization (Adam weight_decay)
batch_size = 128
epochs = 20 # For testing, keep epochs small; increase for better results
k_steps = 10
persistent_chains = True
eval_every = 5

# ---- Sampling ----
num_samples = 16
gibbs_burn_in = 200
sa_start_temp = 10.0
sa_end_temp = 0.1
sa_iterations = 8

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
file_prefix = f"fullmnist_{ARCH_NAME}"

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

def prepare_mnist(dataset, digit_filter=None, threshold=0.5):
    """Extract, optionally filter, and binarize MNIST data."""
    images = dataset.data.float() / 255.0  # (N, 28, 28) in [0,1]
    labels = dataset.targets

    # Filter digits
    if digit_filter is not None:
        mask = torch.zeros(len(labels), dtype=torch.bool)
        for d in digit_filter:
            mask |= (labels == d)
        images = images[mask]
        labels = labels[mask]

    # Binarize
    images = (images >= threshold).float()

    # Flatten to (N, 784)
    images = images.view(images.size(0), -1)

    return images, labels

train_data, train_labels = prepare_mnist(train_dataset, digit_filter, binarize_threshold)
test_data, test_labels   = prepare_mnist(test_dataset,  digit_filter, binarize_threshold)

print(f"  Train: {train_data.shape[0]} samples, {train_data.shape[1]} features")
print(f"  Test:  {test_data.shape[0]} samples, {test_data.shape[1]} features")
print(f"  Digits: {digits_str}")

dataset = TensorDataset(train_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


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
stats = analyze_architecture(G_relabeled, visible_relabeled, hidden_relabeled, ARCH_LABEL)


# =============================================================================
# 5. Model Initialization
# =============================================================================

print("Initializing CustomBoltzmannMachine...")
model = graph_to_bm(G_relabeled, node_labels)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model on {device}, total parameters: {total_params:,}")
print(f"  Visible: {num_visible}, Hidden: {num_hidden}")


# =============================================================================
# 6. Training
# =============================================================================

print(f"\nStarting {train_method} Training (lr={lr}, l2={weight_decay}, epochs={epochs}, k={k_steps})...")
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
    train_data=train_data,
    eval_every=eval_every,
)


# =============================================================================
# 6a. Save model and training history
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

torch.save({
    'model_state_dict': model.state_dict(),
    'training_history': training_history,
    'arch_name': ARCH_NAME,
    'arch_label': ARCH_LABEL,
    'hyperparams': hyperparams,
    'graph_edges': list(G_relabeled.edges()),
    'node_labels': node_labels,
    'hparam_tag': hparam_tag,
}, model_path)
print(f"Saved model to {model_path}")


# =============================================================================
# 6b. Plot training metrics
# =============================================================================

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
             marker='o', markersize=3, linewidth=1.5, color='tab:blue', label='PCD Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    if 'pll' in training_history and len(training_history['pll']) > 0:
        ax1b = ax1.twinx()
        ax1b.set_ylabel("Pseudo Log-Likelihood", color='tab:red')
        ax1b.plot(epochs_range, training_history['pll'],
                  marker='s', markersize=3, linewidth=1.5, color='tab:red', label='PLL')
        ax1b.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title("PCD Loss & PLL")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Reconstruction metrics
    if has_train_recon:
        ax2 = axes[1]
        n_recon = len(training_history['train_recon_mse'])
        recon_epochs = [e for e in range(1, len(training_history['pcd_loss']) + 1)
                        if e % eval_every == 0 or e == len(training_history['pcd_loss'])]
        recon_epochs = recon_epochs[:n_recon]

        ax2.plot(recon_epochs, training_history['train_recon_mse'],
                 marker='o', markersize=3, linewidth=1.5, color='tab:green', label='MSE')
        ax2.plot(recon_epochs, training_history['train_recon_bce'],
                 marker='^', markersize=3, linewidth=1.5, color='tab:orange', label='BCE')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Reconstruction Loss")
        ax2.set_title("Train Reconstruction Metrics")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        ax2b = ax2.twinx()
        ax2b.plot(recon_epochs, training_history['train_recon_acc'],
                  marker='D', markersize=3, linewidth=1.5, color='tab:purple', label='Accuracy')
        ax2b.set_ylabel("Accuracy", color='tab:purple')
        ax2b.tick_params(axis='y', labelcolor='tab:purple')
        ax2b.legend(loc='upper right')

    fig.suptitle(f"Full MNIST {ARCH_LABEL} — Training (K={K}, Hidden={num_hidden}, digits={digits_str})")
    fig.tight_layout()
    plot_path = os.path.join(data_dir, f"{file_prefix}_{hparam_tag}{digit_tag}_training.png")
    fig.savefig(plot_path, dpi=150)
    fig.savefig(f"{file_prefix}_training.png", dpi=150)
    print(f"Saved training plot to {plot_path}")
    plt.close(fig)


# =============================================================================
# 7. Sampling & Visualization (Gibbs)
# =============================================================================

print(f"\nGenerating {num_samples} Gibbs samples (burn-in={gibbs_burn_in})...")
samples = sample_from_bm(model, num_samples=num_samples, burn_in_steps=gibbs_burn_in, method='gibbs')
samples_np = samples.cpu().detach().numpy()

nrows_plot = int(np.ceil(np.sqrt(num_samples)))
ncols_plot = int(np.ceil(num_samples / nrows_plot))

fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(2 * ncols_plot, 2 * nrows_plot))
fig.suptitle(f"Full MNIST {ARCH_LABEL} — Gibbs Samples (K={K}, Ep={epochs}, H={num_hidden})")
for i, ax in enumerate(axes.flat):
    if i < len(samples_np):
        ax.imshow(samples_np[i].reshape(grid_shape), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.tight_layout()
gibbs_path = os.path.join(data_dir, f"{file_prefix}_{hparam_tag}{digit_tag}_samples_gibbs.png")
plt.savefig(gibbs_path, dpi=150)
plt.savefig(f"{file_prefix}_samples_gibbs.png", dpi=150)
print(f"Saved Gibbs samples to {gibbs_path}")
plt.close(fig)


# =============================================================================
# 8. Sampling & Visualization (Simulated Annealing)
# =============================================================================

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

fig_sa, axes_sa = plt.subplots(nrows_plot, ncols_plot, figsize=(2 * ncols_plot, 2 * nrows_plot))
fig_sa.suptitle(f"Full MNIST {ARCH_LABEL} — SA Samples (K={K}, Ep={epochs}, H={num_hidden})")
for i, ax in enumerate(axes_sa.flat):
    if i < len(sa_np):
        ax.imshow(sa_np[i].reshape(grid_shape), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.tight_layout()
sa_path = os.path.join(data_dir, f"{file_prefix}_{hparam_tag}{digit_tag}_samples_sa.png")
plt.savefig(sa_path, dpi=150)
plt.savefig(f"{file_prefix}_samples_sa.png", dpi=150)
print(f"Saved SA samples to {sa_path}")
plt.close(fig_sa)


# =============================================================================
# 9. Test set evaluation
# =============================================================================

print("\nEvaluating on test set...")
from bolmaqua import evaluate_reconstruction, compute_pseudolikelihood

with torch.no_grad():
    test_metrics = evaluate_reconstruction(model, test_data, num_samples=50)
    test_loader_small = DataLoader(TensorDataset(test_data[:500]), batch_size=500)
    test_pll = compute_pseudolikelihood(model, next(iter(test_loader_small))[0], num_samples=50)

print(f"  Test Reconstruction MSE:  {test_metrics['mse']:.4f}")
print(f"  Test Reconstruction BCE:  {test_metrics['bce']:.4f}")
print(f"  Test Reconstruction Acc:  {test_metrics['accuracy']:.4f}")
print(f"  Test PLL (500 samples):   {test_pll:.4f}")

# Save test results alongside model
results_path = os.path.join(data_dir, f"{file_prefix}_{hparam_tag}{digit_tag}_results.txt")
with open(results_path, 'w') as f:
    f.write(f"Full MNIST Test Results — {ARCH_LABEL}\n")
    f.write(f"{'='*60}\n")
    f.write(f"Architecture: {architecture}\n")
    f.write(f"Zephyr K={K}, nodes={n_total}, edges={n_edges}\n")
    f.write(f"Grid shape: {grid_shape}, visible={num_visible}, hidden={num_hidden}\n")
    f.write(f"Digits: {digits_str}\n")
    f.write(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}\n")
    f.write(f"\nHyperparameters:\n")
    for k, v in hyperparams.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"\nTest Results:\n")
    f.write(f"  Reconstruction MSE:  {test_metrics['mse']:.6f}\n")
    f.write(f"  Reconstruction BCE:  {test_metrics['bce']:.6f}\n")
    f.write(f"  Reconstruction Acc:  {test_metrics['accuracy']:.6f}\n")
    f.write(f"  PLL (500 samples):   {test_pll:.6f}\n")
    f.write(f"\nArchitecture Stats:\n")
    f.write(f"  VH edges: {stats['vh']}\n")
    f.write(f"  VV edges: {stats['vv']}\n")
    f.write(f"  HH edges: {stats['hh']}\n")
    f.write(f"\nFiles:\n")
    f.write(f"  Model: {model_path}\n")
    f.write(f"  Hparam tag: {hparam_tag}\n")
print(f"Saved results to {results_path}")


print(f"\n{'='*70}")
print(f"  Full MNIST {ARCH_LABEL} — Complete!")
print(f"  Model: {model_path}")
print(f"{'='*70}")
