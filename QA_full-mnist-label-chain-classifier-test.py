'''
QA-Compatible Full MNIST Classifier on Zephyr Graph with Label Chains

Trains a discriminative Boltzmann Machine on 28x28 MNIST using label CHAINS:
physically-connected groups of Zephyr qubits that represent class labels.

QA compatibility:
  - Every node in the model is a real physical qubit (no contraction/fusion)
  - Label chains are connected subgraphs with ferromagnetic couplings
  - Classification via clamped-pixel sampling + majority vote on chains
  - BQM construction is direct (negate BM coefficients) — no mapping needed

Architecture layout:
  [784 pixel qubits] — [label chain qubits] — [hidden qubits]
  All on the native Zephyr Z(K) topology.
'''

# =============================================================================
#  HYPERPARAMETERS — Edit this section to configure the experiment
# =============================================================================

# -- Architecture --
architecture = "tiling"         # "spectral" (ARCH1), "tiling" (ARCH2), "ilp" (ARCH3)
K = 12                           # Zephyr Z(K) parameter. Z(6)=1248 nodes. Need K>=5 for 28x28.

# -- Data --
grid_shape = (28, 28)           # image size (28x28 for full MNIST)
binarize_threshold = 0.5        # pixel threshold for binary images
digit_filter = None       # None = all 10 digits, or e.g. [0, 1, 2] for a subset

# -- Label Chains (QA-compatible label encoding) --
nodes_per_label = 20             # physical qubits per class chain (greedy/random/degree strategies)
                #note: Zephyr 12 graph diameter is 25, so a bit less than diam seems reasonable
num_seeds_per_label = 5          # seed nodes per class (seed_and_connect strategy only)
chain_strength = 7.0            # ferromagnetic coupling between chain members
chain_mode = 'fixed'            # 'fixed' = clamp after each optimizer step
                                # 'trainable' = let gradients adjust chain couplings
chain_selection_strategy = 'ilp_coverage'       # also: 'random_walk', 'degree_weighted', 'seed_and_connect', 'ilp_coverage'
label_coverage_mode = 'hidden_only'             # or 'hidden_visible'
chain_selection_seed = 42       # reproducibility for chain selection

# -- Classification --
classification_loss_weight = 5.0        # weight for cross-entropy vs CD loss
classification_inference_method = "free_energy"  # 'free_energy', 'mean_field', 'gibbs'

# -- Training --
lr = 5e-3                       # Adam learning rate
weight_decay = 1e-6             # L2 regularization
batch_size = 64
epochs = 100
k_steps = 5                    # CD-k / PCD-k steps
persistent_chains = True        # True = PCD, False = CD
validation_fraction = 0.1       # held-out fraction of MNIST train set
validation_interval = 5         # evaluate train/val accuracy every N epochs
validation_seed = 42            # reproducible train/val split

# -- QA-like Classification Estimate (Neal SA) --
qa_estimate_enabled = True      # run clamped-pixel Neal sampling after training
qa_estimate_num_test_images = 128
qa_estimate_num_reads = 64
qa_estimate_num_sweeps = 100
qa_estimate_beta_range = (0.1, 3.0)
qa_estimate_seed = 42

# -- ARCH1 only: Spectral Bisection --
refinement_iters = 1000
vv_penalty = 0.01              # gamma penalty for VV edges in refinement

# -- ARCH2 only: Hierarchical Tiling --
patch_size = 7                  # must divide both grid dimensions. 7 -> 4x4 = 16 tiles

# -- ARCH3 only: ILP Bidirectional --
ilp_alpha = 1.0                # weight for min VH-degree of visible nodes
ilp_beta  = 1.0                # weight for min VH-degree of hidden nodes
ilp_gamma = 0.01               # penalty per VV edge
ilp_time_limit = 600           # SCIP solver time limit in seconds
gurobi_chain_time_limit = 10*60  # Gurobi time limit for ilp_coverage chain strategy
gurobi_chain_max_candidates = 800  # candidate pool size before connector expansion
gurobi_checkpoint_enabled = True
gurobi_checkpoint_dir = "gurobi_checkpoints"
gurobi_checkpoint_tag = ""     # optional manual tag, e.g. "K12_npl20"

# =============================================================================
#  END OF HYPERPARAMETERS — you should not need to edit below this line
# =============================================================================


#%% Imports

import os
import sys
import time
import torch
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
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

from QA_classifierhelpertest import (
    select_label_chains,
    relabel_with_label_chains,
    get_chain_edge_indices,
    initialize_chain_couplings,
    clamp_chain_couplings,
    prepare_classification_batch,
    compute_class_scores_free_energy,
    compute_chain_offsets,
    train_classifier_bm,
    classify_images_fast,
    evaluate_classifier,
    analyze_architecture,
    analyze_label_chains,
    CHAIN_SELECTION_STRATEGIES,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

_arch1 = import_module('ARCH1-spectral-bisection')
spectral_bisection_partition = _arch1.spectral_bisection_partition
refine_partition_swaps = _arch1.refine_partition_swaps
assign_pixel_order_spatial_arch1 = _arch1.assign_pixel_order_spatial

_arch2 = import_module('ARCH2-hierarchical-tiling')
tiling_assignment = _arch2.tiling_assignment

_arch3 = import_module('ARCH3-ilp-bidirectional')
spectral_warm_start = _arch3.spectral_warm_start
ilp_bidirectional_assignment = _arch3.ilp_bidirectional_assignment
greedy_bidirectional_assignment = _arch3.greedy_bidirectional_assignment
assign_pixel_order_spatial_arch3 = _arch3.assign_pixel_order_spatial
HAS_SCIP = _arch3.HAS_SCIP


# =============================================================================
#  Derived configuration & validation
# =============================================================================

num_visible = grid_shape[0] * grid_shape[1]

if digit_filter is None:
    num_classes = 10
    label_mapping = None
else:
    num_classes = len(digit_filter)
    label_mapping = {digit: i for i, digit in enumerate(sorted(digit_filter))}

if chain_selection_strategy == 'seed_and_connect':
    total_label_nodes_min = num_classes * num_seeds_per_label
    strategy_kwargs = {'num_seeds_per_label': num_seeds_per_label}
elif chain_selection_strategy == 'ilp_coverage':
    total_label_nodes_min = num_classes * nodes_per_label
    strategy_kwargs = {
        'gurobi_time_limit': gurobi_chain_time_limit,
        'max_candidates': gurobi_chain_max_candidates,
        'gurobi_checkpoint_enabled': gurobi_checkpoint_enabled,
        'gurobi_checkpoint_dir': gurobi_checkpoint_dir,
    }
    if gurobi_checkpoint_tag:
        strategy_kwargs['gurobi_checkpoint_tag'] = gurobi_checkpoint_tag
else:
    total_label_nodes_min = num_classes * nodes_per_label
    strategy_kwargs = {}

n_zephyr = 16 * K * (2 * K + 1)

# --- Validation ---
VALID_ARCHITECTURES = {"spectral", "tiling", "ilp"}
if architecture not in VALID_ARCHITECTURES:
    raise ValueError(f"Unknown architecture '{architecture}'. Choose from: {VALID_ARCHITECTURES}")

if n_zephyr < num_visible + total_label_nodes_min:
    raise ValueError(
        f"Zephyr Z({K}) has only {n_zephyr} nodes, but need at least "
        f"{num_visible} pixels + {total_label_nodes_min} label nodes. Increase K."
    )

if architecture == "tiling":
    if grid_shape[0] % patch_size != 0 or grid_shape[1] % patch_size != 0:
        raise ValueError(
            f"patch_size={patch_size} does not divide grid_shape={grid_shape}."
        )

# --- Tags for filenames ---
ARCH_NAMES = {
    "spectral": ("spectral", "ARCH1: Spectral Bisection"),
    "tiling":   ("tiling",   "ARCH2: Hierarchical Tiling"),
    "ilp":      ("ilp",      "ARCH3: ILP Bidirectional"),
}
ARCH_NAME, ARCH_LABEL = ARCH_NAMES[architecture]
train_method = "PCD" if persistent_chains else "CD"

hparam_base = f"{train_method}_K{K}_lr{lr}_l2{weight_decay}_bs{batch_size}_ep{epochs}_k{k_steps}"
if architecture == "spectral":
    hparam_arch = f"_refine{refinement_iters}_gamma{vv_penalty}"
elif architecture == "tiling":
    hparam_arch = f"_patch{patch_size}"
elif architecture == "ilp":
    hparam_arch = f"_a{ilp_alpha}_b{ilp_beta}_g{ilp_gamma}"
hparam_tag = hparam_base + hparam_arch

if digit_filter is not None:
    digits_str = "".join(str(d) for d in sorted(digit_filter))
    digit_tag = f"_digits{digits_str}"
else:
    digits_str = "all"
    digit_tag = "_digitsAll"

file_prefix = f"QA_fullmnist_classifier_{ARCH_NAME}"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)


# =============================================================================
#  Helper: Neal conditional sampling for QA estimate
# =============================================================================

def estimate_classifier_with_neal_conditional_sampling(
    model,
    pixels,
    labels,
    num_pixel_nodes,
    num_classes,
    label_chains,
    num_reads=64,
    num_sweeps=100,
    beta_range=(0.1, 3.0),
    seed=None,
):
    """
    QA-deployable classification: clamp pixel qubits, sample label+hidden
    with Neal SA, decode label chains via majority vote.

    This is the closest classical simulation of what happens on real QA:
      1. Build BQM from BM weights (negate coefficients)
      2. Fix pixel variables to image values
      3. Sample free variables (label chains + hidden) via SA
      4. For each read, decode each class's chain via majority vote
      5. Predict class with highest mean chain activation across reads
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

    base_bqm = dimod.BinaryQuadraticModel(
        linear, quadratic, 0.0, vartype=dimod.BINARY,
    )
    sampler = neal.SimulatedAnnealingSampler()

    all_preds = []
    all_labels = []
    all_confidences = []
    chain_offsets = compute_chain_offsets(label_chains)
    total_label_qubits = chain_offsets[-1]
    chain_lengths = [len(c) for c in label_chains]

    print(f"\n{'='*70}")
    print("  QA-LIKE CONDITIONAL SAMPLING ESTIMATE (NEAL)")
    print(f"{'='*70}")
    print(f"Evaluating {len(labels)} test images with clamped-pixel Neal sampling...")
    if len(set(chain_lengths)) == 1:
        print(f"  Label chain: {chain_lengths[0]} qubits/class, "
              f"majority vote decoding")
    else:
        print(f"  Label chains: {min(chain_lengths)}-{max(chain_lengths)} "
              f"qubits/class (variable), majority vote decoding")
    print(f"  num_reads={num_reads}, num_sweeps={num_sweeps}, "
          f"beta_range={beta_range}")

    for idx in range(len(labels)):
        pixel_vec = pixels[idx].cpu().numpy().astype(np.int8)
        bqm = base_bqm.copy()

        for pixel_idx in range(num_pixel_nodes):
            bqm.fix_variable(pixel_idx, int(pixel_vec[pixel_idx]))

        sampleset = sampler.sample(
            bqm, num_reads=num_reads, num_sweeps=num_sweeps,
            beta_range=beta_range, seed=seed,
        )
        var_list = list(sampleset.variables)

        class_scores = np.zeros(num_classes, dtype=np.float32)
        for class_idx in range(num_classes):
            chain_start = num_pixel_nodes + chain_offsets[class_idx]
            chain_length = len(label_chains[class_idx])
            chain_activations = np.zeros(num_reads, dtype=np.float32)

            for node_offset in range(chain_length):
                label_qubit_idx = chain_start + node_offset
                if label_qubit_idx in var_list:
                    col = var_list.index(label_qubit_idx)
                    chain_activations += sampleset.record.sample[:, col]

            class_scores[class_idx] = chain_activations.mean() / chain_length

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
            per_class_acc[class_idx] = float(
                np.mean(all_preds[class_mask] == all_labels[class_mask])
            )
        else:
            per_class_acc[class_idx] = 0.0

    print(f"\n  QA-like Neal Accuracy: {100*accuracy:.2f}%")
    print(f"  Mean confidence: {100*np.mean(all_confidences):.2f}%")
    for class_idx in range(num_classes):
        print(f"  Class {class_idx} Accuracy: "
              f"{100*per_class_acc[class_idx]:.2f}%")

    return {
        'accuracy': accuracy,
        'per_class': per_class_acc,
        'mean_confidence': float(np.mean(all_confidences)),
        'num_eval_images': int(len(labels)),
        'num_reads': int(num_reads),
        'num_sweeps': int(num_sweeps),
        'beta_range': beta_range,
        'chain_lengths': chain_lengths,
    }


def prepare_mnist(dataset, digit_filter=None, threshold=0.5, label_mapping=None):
    """Extract, optionally filter, and binarize MNIST data."""
    images = dataset.data.float() / 255.0
    labels = dataset.targets

    if digit_filter is not None:
        mask = torch.zeros(len(labels), dtype=torch.bool)
        for d in digit_filter:
            mask |= (labels == d)
        images = images[mask]
        labels = labels[mask]
        if label_mapping is not None:
            labels = torch.tensor(
                [label_mapping[l.item()] for l in labels], dtype=torch.long,
            )

    images = (images >= threshold).float()
    images = images.view(images.size(0), -1)
    return images, labels


# =============================================================================
#  Print configuration summary
# =============================================================================

print("=" * 70)
print(f"  QA-COMPATIBLE FULL MNIST CLASSIFIER — {ARCH_LABEL}")
print(f"  Zephyr Z({K}): {n_zephyr} nodes for {grid_shape[0]}x{grid_shape[1]} images")
print("=" * 70)
print(f"  Architecture:   {architecture}")
print(f"  Grid shape:     {grid_shape} ({num_visible} pixel qubits)")
print(f"  Digits:         {digits_str}")
print(f"  Classes:        {num_classes}")
if chain_selection_strategy == 'seed_and_connect':
    print(f"  Label chains:   {nodes_per_label} qubits/class (2-hop coverage guided), "
          f"strategy={chain_selection_strategy}")
elif chain_selection_strategy == 'ilp_coverage':
    print(f"  Label chains:   {nodes_per_label} qubits/class (Gurobi max-min, "
          f"time_limit={gurobi_chain_time_limit}s), strategy={chain_selection_strategy}")
    print(f"  Gurobi ckpt:    enabled={gurobi_checkpoint_enabled}, "
          f"dir={gurobi_checkpoint_dir}, max_candidates={gurobi_chain_max_candidates}")
else:
    print(f"  Label chains:   {nodes_per_label} qubits/class, "
          f"strategy={chain_selection_strategy}")
print(f"  Chain coupling: strength={chain_strength}, mode={chain_mode}")
print(f"  Training:       lr={lr}, l2={weight_decay}, epochs={epochs}, "
      f"k={k_steps}, bs={batch_size}, {train_method}")
print(f"  Validation:     fraction={validation_fraction}, "
    f"interval={validation_interval} epochs")
print(f"  Cls loss wt:    {classification_loss_weight}")
print(f"  Inference:      {classification_inference_method}")
print(f"  QA estimate:    enabled={qa_estimate_enabled}, "
      f"images={qa_estimate_num_test_images}, "
      f"reads={qa_estimate_num_reads}, sweeps={qa_estimate_num_sweeps}")
if architecture == "spectral":
    print(f"  ARCH1 params:   refine={refinement_iters}, gamma={vv_penalty}")
elif architecture == "tiling":
    print(f"  ARCH2 params:   patch_size={patch_size}")
elif architecture == "ilp":
    print(f"  ARCH3 params:   alpha={ilp_alpha}, beta={ilp_beta}, "
          f"gamma={ilp_gamma}, time_limit={ilp_time_limit}s")
    print(f"  SCIP available: {HAS_SCIP}")
print(f"  Strategies:     {list(CHAIN_SELECTION_STRATEGIES.keys())}")
print(f"  Hparam tag:     {hparam_tag}")
print("=" * 70)


# =============================================================================
#  1. Data Loading
# =============================================================================

print("\nLoading full MNIST dataset from torchvision...")

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor(),
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor(),
)

train_data, train_labels = prepare_mnist(
    train_dataset, digit_filter, binarize_threshold, label_mapping,
)
test_data, test_labels = prepare_mnist(
    test_dataset, digit_filter, binarize_threshold, label_mapping,
)

if validation_fraction > 0.0:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    split_generator = torch.Generator().manual_seed(validation_seed)
    shuffled_indices = torch.randperm(len(train_data), generator=split_generator)
    val_count = max(1, int(round(validation_fraction * len(train_data))))
    val_indices = shuffled_indices[:val_count]
    train_indices = shuffled_indices[val_count:]
    val_data = train_data[val_indices]
    val_labels = train_labels[val_indices]
    train_data = train_data[train_indices]
    train_labels = train_labels[train_indices]
else:
    val_data = train_data[:0]
    val_labels = train_labels[:0]

print(f"  Train: {train_data.shape[0]} samples, {train_data.shape[1]} features")
print(f"  Val:   {val_data.shape[0]} samples, {val_data.shape[1]} features")
print(f"  Test:  {test_data.shape[0]} samples, {test_data.shape[1]} features")

train_dataset_final = TensorDataset(train_data, train_labels)
val_dataset_final = TensorDataset(val_data, val_labels)
test_dataset_final = TensorDataset(test_data, test_labels)

loader = DataLoader(
    train_dataset_final, batch_size=batch_size, shuffle=True, drop_last=True,
)
train_eval_loader = DataLoader(
    train_dataset_final, batch_size=batch_size, shuffle=False,
)
val_loader = DataLoader(
    val_dataset_final, batch_size=batch_size, shuffle=False,
)
test_loader = DataLoader(
    test_dataset_final, batch_size=batch_size, shuffle=False,
)


# =============================================================================
#  2. Architecture Construction (pixel/hidden partition)
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

if architecture == "spectral":
    print(f"\n--- ARCH1: Spectral Bisection ---")
    visible_set = spectral_bisection_partition(G_zephyr, num_visible)
    visible_set = refine_partition_swaps(
        G_zephyr, visible_set, num_visible,
        max_iters=refinement_iters, gamma=vv_penalty,
    )
    visible_in_pixel_order = assign_pixel_order_spatial_arch1(
        G_zephyr, visible_set, grid_shape,
    )
    hidden_nodes = [n for n in sorted(G_zephyr.nodes()) if n not in visible_set]

elif architecture == "tiling":
    print(f"\n--- ARCH2: Hierarchical Tiling (patch={patch_size}x{patch_size}) ---")
    visible_in_pixel_order, hidden_nodes = tiling_assignment(
        G_zephyr, grid_shape, patch_size,
    )

elif architecture == "ilp":
    print(f"\n--- ARCH3: ILP Bidirectional ---")
    warm_start = spectral_warm_start(G_zephyr, num_visible)
    if HAS_SCIP:
        visible_set, ilp_stats = ilp_bidirectional_assignment(
            G_zephyr, num_visible,
            alpha=ilp_alpha, beta=ilp_beta, gamma=ilp_gamma,
            time_limit=ilp_time_limit, warm_start_set=warm_start,
        )
        if visible_set is None:
            print("  ILP failed — falling back to greedy")
            visible_set = greedy_bidirectional_assignment(
                G_zephyr, num_visible, gamma=ilp_gamma,
            )
    else:
        print("  Greedy bidirectional (SCIP not available)...")
        visible_set = greedy_bidirectional_assignment(
            G_zephyr, num_visible, gamma=ilp_gamma,
        )
    visible_in_pixel_order = assign_pixel_order_spatial_arch3(
        G_zephyr, visible_set, grid_shape,
    )
    hidden_nodes = [n for n in sorted(G_zephyr.nodes()) if n not in visible_set]

num_hidden_initial = len(hidden_nodes)

print(f"\nRelabeling graph (pixels=0..{num_visible-1}, "
      f"hidden={num_visible}..{num_visible+num_hidden_initial-1})...")
G_relabeled, mapping_initial = relabel_visible_first(
    G_zephyr, visible_in_pixel_order,
)

node_labels_initial = {}
for i in range(num_visible):
    node_labels_initial[i] = 'visible'
for i in range(num_visible, num_visible + num_hidden_initial):
    node_labels_initial[i] = 'hidden'

t_arch = time.time() - t0
print(f"Architecture construction time: {t_arch:.1f}s")


# =============================================================================
#  3. Label Chain Selection & Relabeling
# =============================================================================

visible_relabeled = list(range(num_visible))
hidden_relabeled = list(range(num_visible, num_visible + num_hidden_initial))

stats = analyze_architecture(
    G_relabeled, visible_relabeled, hidden_relabeled,
    f"{ARCH_LABEL} (base partition, before label chains)",
)

print(f"\n--- Selecting Label Chains ({chain_selection_strategy}) ---")

label_chains_original, chain_stats = select_label_chains(
    G_relabeled,
    visible_relabeled,
    hidden_relabeled,
    num_classes=num_classes,
    nodes_per_label=nodes_per_label,
    strategy=chain_selection_strategy,
    coverage_mode=label_coverage_mode,
    seed=chain_selection_seed,
    verbose=True,
    strategy_kwargs=strategy_kwargs,
)

G_final, label_chains, remaining_hidden, node_labels, chain_mapping = (
    relabel_with_label_chains(
        G_relabeled,
        visible_relabeled,
        hidden_relabeled,
        label_chains_original,
    )
)

total_label_nodes = sum(len(c) for c in label_chains)
num_visible_total = num_visible + total_label_nodes
num_hidden = len(remaining_hidden)

chain_lengths = [len(c) for c in label_chains]
if len(set(chain_lengths)) == 1:
    chain_desc = f"{num_classes} chains x {chain_lengths[0]}"
else:
    chain_desc = (f"{num_classes} chains, "
                  f"lengths {min(chain_lengths)}-{max(chain_lengths)}")

print(f"\n  Final architecture (QA-compatible):")
print(f"    Pixel qubits:       {num_visible}")
print(f"    Label chain qubits: {total_label_nodes} ({chain_desc})")
print(f"    Hidden qubits:      {num_hidden}")
print(f"    Total nodes:        {G_final.number_of_nodes()} (all physical)")

analyze_label_chains(
    G_final, label_chains, num_visible, num_classes,
)

all_visible = list(range(num_visible_total))
stats_final = analyze_architecture(
    G_final, all_visible, remaining_hidden,
    f"{ARCH_LABEL} (with label chains)",
)


# =============================================================================
#  4. Model Initialization + Chain Couplings
# =============================================================================

print("Initializing CustomBoltzmannMachine...")
model = graph_to_bm(G_final, node_labels)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model on {device}, total parameters: {total_params:,}")
print(f"  Visible: {num_visible_total}, Hidden: {num_hidden}")

chain_edges = get_chain_edge_indices(G_final, label_chains)
print(f"\nInitializing chain couplings (mode={chain_mode})...")
initialize_chain_couplings(model, chain_edges, chain_strength)


# =============================================================================
#  5. Training
# =============================================================================

optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, weight_decay=weight_decay,
)

training_history = train_classifier_bm(
    model, loader, optimizer,
    num_epochs=epochs, k_steps=k_steps,
    label_chains=label_chains,
    num_classes=num_classes,
    nodes_per_label=nodes_per_label,
    classification_loss_weight=classification_loss_weight,
    chain_edges=chain_edges,
    chain_strength=chain_strength,
    chain_mode=chain_mode,
    train_eval_loader=train_eval_loader,
    val_loader=val_loader if len(val_dataset_final) > 0 else None,
    validation_interval=validation_interval,
    validation_inference_method=classification_inference_method,
    validation_num_gibbs_steps=10,
)


# =============================================================================
#  6. Classification Evaluation
# =============================================================================

print(f"\n{'='*70}")
print("  CLASSIFICATION EVALUATION")
print(f"{'='*70}")

model.eval()
with torch.inference_mode():
    train_acc, train_per_class = evaluate_classifier(
        model,
        DataLoader(train_dataset_final, batch_size=batch_size, shuffle=False),
        label_chains,
        num_classes=num_classes,
        nodes_per_label=nodes_per_label,
        num_gibbs_steps=10,
        inference_method=classification_inference_method,
    )

    test_acc, test_per_class = evaluate_classifier(
        model, test_loader,
        label_chains,
        num_classes=num_classes,
        nodes_per_label=nodes_per_label,
        num_gibbs_steps=10,
        inference_method=classification_inference_method,
    )


# =============================================================================
#  7. Visualize Predictions
# =============================================================================

print("\nVisualizing predictions on test images...")
sample_batch = next(iter(test_loader))
sample_pixels, sample_labels = sample_batch[0][:9].to(device), sample_batch[1][:9]

with torch.inference_mode():
    preds, confs = classify_images_fast(
        model, sample_pixels,
        label_chains,
        num_classes=num_classes,
        nodes_per_label=nodes_per_label,
        num_gibbs_steps=50,
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
    ax.set_title(
        f"True: {true_label}, Pred: {pred_label}\nConf: {conf:.2f}",
        color=color,
    )
    ax.axis('off')

fig.suptitle(
    f"QA-Compatible Classifier {ARCH_LABEL} — "
    f"Test Predictions (Acc={100*test_acc:.1f}%)"
)
plt.tight_layout()
pred_path = os.path.join(
    data_dir,
    f"QA_classifier_{ARCH_NAME}_{hparam_tag}_predictions.png",
)
plt.savefig(pred_path)
print(f"Saved predictions to {pred_path}")
plt.show()

print(f"\n{'='*60}")
print(f"CLASSIFICATION RESULTS")
print(f"{'='*60}")
print(f"Train Accuracy: {100*train_acc:.2f}%")
print(f"Test Accuracy:  {100*test_acc:.2f}%")
if training_history.get('val_accuracy'):
    print(f"Best Val Accuracy: {100*max(training_history['val_accuracy']):.2f}%")
print(f"Per-class test accuracy:")
for digit, acc in test_per_class.items():
    print(f"  Digit {digit}: {100*acc:.2f}%")
print(f"{'='*60}\n")

if training_history.get('validation_epochs'):
    fig, ax = plt.subplots(figsize=(8, 5))
    val_epochs = training_history['validation_epochs']
    train_acc_curve = [100 * acc for acc in training_history['train_accuracy']]
    val_acc_curve = [100 * acc for acc in training_history['val_accuracy']]
    ax.plot(val_epochs, train_acc_curve, marker='o', label='Train')
    ax.plot(val_epochs, val_acc_curve, marker='o', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Train vs Validation Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
#  8. QA-like Classification via Neal Conditional Sampling
# =============================================================================

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
        label_chains=label_chains,
        num_reads=qa_estimate_num_reads,
        num_sweeps=qa_estimate_num_sweeps,
        beta_range=qa_estimate_beta_range,
        seed=qa_estimate_seed,
    )


# =============================================================================
#  9. Save model checkpoint
# =============================================================================

classification_results = {
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'train_per_class': train_per_class,
    'test_per_class': test_per_class,
    'qa_like_neal_estimate': qa_like_results,
}

model_path = os.path.join(
    data_dir,
    f"{file_prefix}_{hparam_tag}{digit_tag}_model.pt",
)

hyperparams = {
    'architecture': architecture,
    'K': K, 'grid_shape': grid_shape,
    'lr': lr, 'weight_decay': weight_decay,
    'batch_size': batch_size, 'epochs': epochs, 'k_steps': k_steps,
    'persistent_chains': persistent_chains,
    'binarize_threshold': binarize_threshold,
    'digit_filter': digit_filter,
    'num_visible': num_visible,
    'num_visible_total': num_visible_total,
    'num_hidden': num_hidden,
    'num_train_samples': len(train_data),
    'num_val_samples': len(val_data),
    'num_test_samples': len(test_data),
    'nodes_per_label': nodes_per_label,
    'num_seeds_per_label': num_seeds_per_label if chain_selection_strategy == 'seed_and_connect' else None,
    'gurobi_chain_time_limit': gurobi_chain_time_limit if chain_selection_strategy == 'ilp_coverage' else None,
    'gurobi_chain_max_candidates': gurobi_chain_max_candidates if chain_selection_strategy == 'ilp_coverage' else None,
    'gurobi_checkpoint_enabled': gurobi_checkpoint_enabled if chain_selection_strategy == 'ilp_coverage' else None,
    'gurobi_checkpoint_dir': gurobi_checkpoint_dir if chain_selection_strategy == 'ilp_coverage' else None,
    'gurobi_checkpoint_tag': gurobi_checkpoint_tag if chain_selection_strategy == 'ilp_coverage' else None,
    'total_label_nodes': total_label_nodes,
    'chain_lengths': [len(c) for c in label_chains],
    'chain_strength': chain_strength,
    'chain_mode': chain_mode,
    'chain_selection_strategy': chain_selection_strategy,
    'label_coverage_mode': label_coverage_mode,
    'chain_selection_seed': chain_selection_seed,
    'classification_loss_weight': classification_loss_weight,
    'classification_inference_method': classification_inference_method,
    'validation_fraction': validation_fraction,
    'validation_interval': validation_interval,
    'validation_seed': validation_seed,
    'chain_stats': chain_stats,
    'label_chains_relabeled': [list(c) for c in label_chains],
    'chain_edges': chain_edges,
    'qa_estimate_enabled': qa_estimate_enabled,
    'qa_estimate_num_test_images': qa_estimate_num_test_images,
    'qa_estimate_num_reads': qa_estimate_num_reads,
    'qa_estimate_num_sweeps': qa_estimate_num_sweeps,
    'qa_estimate_beta_range': qa_estimate_beta_range,
}

if architecture == "spectral":
    hyperparams.update({
        'refinement_iters': refinement_iters,
        'vv_penalty': vv_penalty,
    })
elif architecture == "tiling":
    hyperparams.update({'patch_size': patch_size})
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
    'graph_edges': list(G_final.edges()),
    'node_labels': node_labels,
    'hparam_tag': hparam_tag,
    'classification_results': classification_results,
}, model_path)
print(f"\nSaved model checkpoint to {model_path}")


