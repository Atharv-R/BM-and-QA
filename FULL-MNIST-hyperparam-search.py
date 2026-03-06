"""
FULL-MNIST-hyperparam-search.py — Random hyperparameter search for BMs on full MNIST

Runs a time-budgeted random search over architectures and training hyperparameters.
Each trial:
  1. Samples a random hyperparameter config (architecture + training params)
  2. Constructs the architecture on Zephyr Z(K)
  3. Trains the BM with PCD
  4. Evaluates sample quality via FID (using eval_quality.py)
  5. Updates the top-5 leaderboard JSON + saves top-5 model checkpoints

The leaderboard is saved to:
    data/{datetime}_hyperparam_best_results.json
and is updated after every trial, so results are never lost even if interrupted.

Usage:
    python FULL-MNIST-hyperparam-search.py                  # Run with defaults
    python FULL-MNIST-hyperparam-search.py --hours 12       # Run for 12 hours
    python FULL-MNIST-hyperparam-search.py --max-trials 50  # Run up to 50 trials
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from copy import deepcopy

import torch
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import dwave_networkx as dnx

from bolmaqua import (
    graph_to_bm,
    train_boltzmann_machine_pcd,
    device,
    relabel_visible_first,
)

from eval_quality import (
    eval_samples_fullMNIST,
    train_mnist_classifier,
    MNISTQualityClassifier,
    _compute_features,
)

# Architecture imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

_arch1 = import_module("ARCH1-spectral-bisection")
spectral_bisection_partition = _arch1.spectral_bisection_partition
refine_partition_swaps = _arch1.refine_partition_swaps
assign_pixel_order_spatial_arch1 = _arch1.assign_pixel_order_spatial
analyze_architecture = _arch1.analyze_architecture

_arch2 = import_module("ARCH2-hierarchical-tiling")
tiling_assignment = _arch2.tiling_assignment

_arch3 = import_module("ARCH3-ilp-bidirectional")
spectral_warm_start = _arch3.spectral_warm_start
ilp_bidirectional_assignment = _arch3.ilp_bidirectional_assignment
greedy_bidirectional_assignment = _arch3.greedy_bidirectional_assignment
assign_pixel_order_spatial_arch3 = _arch3.assign_pixel_order_spatial
HAS_SCIP = _arch3.HAS_SCIP

_arch4 = import_module("ARCH4-embedding-hybrid")
embedding_hybrid_assignment = _arch4.embedding_hybrid_assignment
get_arch4_cache_path = _arch4.get_arch4_cache_path


# =============================================================================
# CLI Arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Hyperparameter search for BMs on full MNIST")
parser.add_argument("--hours", type=float, default=8.0,
                    help="Time budget in hours (default: 8)")
parser.add_argument("--max-trials", type=int, default=999,
                    help="Maximum number of trials (default: 999)")
parser.add_argument("--K", type=int, default=8,
                    help="Zephyr graph parameter K (default: 8)")
parser.add_argument("--top-n", type=int, default=5,
                    help="Number of best configs to track (default: 5)")
parser.add_argument("--eval-samples", type=int, default=400,
                    help="Samples per sampling method for FID eval (default: 400)")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed (default: None = random)")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to existing leaderboard JSON to resume from")
args = parser.parse_args()


# =============================================================================
# Fixed Config
# =============================================================================

GRID_SHAPE = (28, 28)
NUM_VISIBLE = GRID_SHAPE[0] * GRID_SHAPE[1]  # 784
BINARIZE_THRESHOLD = 0.5
K = args.K
N_ZEPHYR = 16 * K * (2 * K + 1)
DATA_DIR = "data"
TOP_N = args.top_n
EVAL_NUM_SAMPLES = args.eval_samples
TIME_BUDGET_SECONDS = args.hours * 3600
MAX_TRIALS = args.max_trials

# Eval: use Gibbs, batched SA, and Neal (D-Wave SA emulator)
EVAL_METHODS = ["gibbs", "sa_batched", "neal"]

# Architecture pool — only these will be sampled and pre-computed
# Change this list to control which architectures are searched over.
#ARCH_POOL = ["spectral", "tiling", "ilp", "embedding"]  # all four
ARCH_POOL = ["spectral", "tiling", "embedding"]


# Validate K
if N_ZEPHYR < NUM_VISIBLE:
    raise ValueError(
        f"Zephyr Z({K}) has only {N_ZEPHYR} nodes, need >= {NUM_VISIBLE}. Increase K."
    )

os.makedirs(DATA_DIR, exist_ok=True)

# Timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LEADERBOARD_PATH = os.path.join(DATA_DIR, f"{RUN_TIMESTAMP}_hyperparam_best_results.json")

# Seed
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# =============================================================================
# Search Space Definition
# =============================================================================

def sample_hyperparams() -> dict:
    """
    Sample a random hyperparameter configuration.

    Returns a dict with all params needed to run one trial.
    """
    # Architecture (sampled from ARCH_POOL)
    architecture = random.choice(ARCH_POOL)

    # Learning rate: log-uniform from 1e-5 to 1e-2
    lr = 10 ** random.uniform(-5, -2)

    # Weight decay: log-uniform from 1e-8 to 1e-5
    weight_decay = 10 ** random.uniform(-8, -5)

    # Batch size: from a set of powers of 2
    batch_size = random.choice([32, 64, 128, 256])

    # Epochs: moderate range for search (not too long)
    epochs = random.choice([15, 20, 30, 40, 50])

    # CD/PCD k-steps: important for mixing
    k_steps = random.choice([5, 10, 15, 20, 25])

    # PCD vs CD: PCD is generally better, but include CD occasionally
    persistent = random.choices([True, False], weights=[0.85, 0.15])[0]

    config = {
        "architecture": architecture,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "k_steps": k_steps,
        "persistent": persistent,
        "K": K,
        "grid_shape": list(GRID_SHAPE),
        "num_visible": NUM_VISIBLE,
    }

    # Architecture-specific parameters
    if architecture == "spectral":
        config["refinement_iters"] = random.choice([500, 1000, 2000])
        config["vv_penalty"] = 10 ** random.uniform(-3, -1)  # 0.001 to 0.1

    elif architecture == "tiling":
        # patch_size must divide 28. Valid: 1, 2, 4, 7, 14
        # Skip 1 (trivial), 14 (too few tiles), 28 (single tile)
        config["patch_size"] = random.choice([2, 4, 7])

    elif architecture == "ilp":
        pass  # ILP uses a single pre-solved canonical assignment (no arch params to tune)

    elif architecture == "embedding":
        config["initial_grid_size"] = random.choice([8, 9, 10])
        config["add_node_criterion"] = random.choice(["graph_distance", "connectivity"])

    return config


# =============================================================================
# Data Loading (done once)
# =============================================================================

print("=" * 70)
print(f"  FULL MNIST HYPERPARAMETER SEARCH")
print(f"  Zephyr Z({K}): {N_ZEPHYR} nodes for {GRID_SHAPE[0]}x{GRID_SHAPE[1]} images")
print(f"  Time budget: {args.hours}h | Max trials: {MAX_TRIALS}")
print(f"  Eval samples: {EVAL_NUM_SAMPLES} per method | Top-{TOP_N} tracked")
print(f"  Leaderboard: {LEADERBOARD_PATH}")
print("=" * 70)

print("\nLoading full MNIST dataset...")
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())


def prepare_mnist(dataset, threshold=0.5):
    images = dataset.data.float() / 255.0
    images = (images >= threshold).float()
    images = images.view(images.size(0), -1)
    labels = dataset.targets
    return images, labels


train_data, train_labels = prepare_mnist(train_dataset, BINARIZE_THRESHOLD)
test_data, test_labels = prepare_mnist(test_dataset, BINARIZE_THRESHOLD)
print(f"  Train: {train_data.shape[0]}, Test: {test_data.shape[0]}")

# Pre-train the classifier (reused across all trials)
print("\nPreparing quality classifier...")
classifier = train_mnist_classifier()

# Pre-compute real data features (reused across all trials)
print("Pre-computing real data features for FID...")
n_real_ref = min(10000, len(train_data))
real_ref_idx = torch.randperm(len(train_data))[:n_real_ref]
real_ref_data = train_data[real_ref_idx]
real_features_cache = _compute_features(classifier, real_ref_data)
print(f"  Real features cached: {real_features_cache.shape}")

# Pre-generate the Zephyr graph (reused across all trials)
print(f"\nGenerating Zephyr graph Z({K})...")
G_zephyr = dnx.zephyr_graph(K)
print(f"  Nodes: {G_zephyr.number_of_nodes()}, Edges: {G_zephyr.number_of_edges()}")


# =============================================================================
# Pre-compute & cache architecture assignments
# =============================================================================
# Tiling is fully deterministic for a given patch_size — pre-compute all 3.
# ILP is expensive (~minutes to hours) — solve once with canonical params and
#   cache to disk. For a fixed K, the V/H partition doesn't change, so only
#   training hyperparams need searching.
# Spectral has random refinement swaps, so it must be recomputed each trial.

_arch_assignment_cache = {}  # key -> (visible_in_pixel_order, hidden_nodes)

# --- Tiling: pre-compute for all patch sizes ---
if "tiling" in ARCH_POOL:
    print("\nPre-computing tiling assignments...")
    for _ps in [2, 4, 7]:
        _t0 = time.time()
        _vis, _hid = tiling_assignment(G_zephyr, GRID_SHAPE, _ps)
        _arch_assignment_cache[f"tiling_patch{_ps}"] = (_vis, _hid)
        print(f"  patch_size={_ps}: {len(_hid)} hidden nodes  ({time.time()-_t0:.1f}s)")
else:
    print("\nSkipping tiling pre-computation (not in ARCH_POOL)")

# --- ILP: solve once with canonical params, cache to disk ---
ILP_CACHE_PATH = os.path.join(DATA_DIR, f"ilp_canonical_assignment_K{K}.json")

if "ilp" not in ARCH_POOL:
    print("\nSkipping ILP pre-computation (not in ARCH_POOL)")
elif os.path.exists(ILP_CACHE_PATH):
    print(f"\nLoading cached ILP assignment from {ILP_CACHE_PATH}...")
    with open(ILP_CACHE_PATH, "r") as f:
        _ilp_data = json.load(f)
    _ilp_vis = _ilp_data["visible_in_pixel_order"]
    _ilp_hid = _ilp_data["hidden_nodes"]
    print(f"  Loaded: {len(_ilp_vis)} visible, {len(_ilp_hid)} hidden nodes")
    _arch_assignment_cache["ilp_canonical"] = (_ilp_vis, _ilp_hid)
else:
    print(f"\nNo cached ILP assignment found at {ILP_CACHE_PATH}.")
    print(f"  Solving ILP with canonical params (alpha=1, beta=1, gamma=0.01, up to 1 hour)...")
    print(f"  This is a one-time cost; the result will be saved for future runs.")
    _warm_start = spectral_warm_start(G_zephyr, NUM_VISIBLE)
    _ilp_visible_set = None
    if HAS_SCIP:
        _ilp_visible_set, _ilp_info = ilp_bidirectional_assignment(
            G_zephyr, NUM_VISIBLE,
            alpha=1.0, beta=1.0, gamma=0.05,
            time_limit=3600,  # 1 hour
            warm_start_set=_warm_start,
        )
    if _ilp_visible_set is None:
        print("  ILP returned no solution (or SCIP unavailable), falling back to greedy...")
        _ilp_visible_set = greedy_bidirectional_assignment(
            G_zephyr, NUM_VISIBLE, gamma=0.05
        )
    _ilp_vis = assign_pixel_order_spatial_arch3(G_zephyr, _ilp_visible_set, GRID_SHAPE)
    _ilp_hid = [n for n in sorted(G_zephyr.nodes()) if n not in _ilp_visible_set]

    # Save to disk for future runs
    with open(ILP_CACHE_PATH, "w") as f:
        json.dump({
            "visible_in_pixel_order": _ilp_vis,
            "hidden_nodes": _ilp_hid,
            "K": K,
            "num_visible": NUM_VISIBLE,
            "grid_shape": list(GRID_SHAPE),
            "solver_params": {"alpha": 1.0, "beta": 1.0, "gamma": 0.05, "time_limit": 3600},
        }, f, indent=2)
    print(f"  Saved ILP assignment to {ILP_CACHE_PATH}")
    _arch_assignment_cache["ilp_canonical"] = (_ilp_vis, _ilp_hid)

# --- Embedding (ARCH4): pre-compute for all candidate initial_grid_sizes ---
if "embedding" not in ARCH_POOL:
    print("\nSkipping embedding pre-computation (not in ARCH_POOL)")
else:
    print("\nPre-computing embedding (ARCH4) assignments...")
    _embedding_grid_sizes = [8, 9]
    _embedding_criteria = ["graph_distance", "connectivity"]
    for _igs in _embedding_grid_sizes:
        for _crit in _embedding_criteria:
            _cache_key = f"embedding_init{_igs}_{_crit}"
            _cache_path = get_arch4_cache_path(K, GRID_SHAPE, _igs, _crit, DATA_DIR)
            if os.path.exists(_cache_path):
                with open(_cache_path, "r") as f:
                    _emb_data = json.load(f)
                _emb_vis = _emb_data["visible_in_pixel_order"]
                _emb_hid = _emb_data["hidden_nodes"]
                _arch_assignment_cache[_cache_key] = (_emb_vis, _emb_hid)
                print(f"  init={_igs}, {_crit}: loaded from cache ({len(_emb_hid)} hidden)")
            else:
                _t0 = time.time()
                _emb_vis, _emb_hid = embedding_hybrid_assignment(
                    G_zephyr, GRID_SHAPE,
                    initial_grid_size=_igs,
                    add_node_criterion=_crit,
                    K=K,
                    data_dir=DATA_DIR,
                )
                if _emb_vis is not None:
                    _arch_assignment_cache[_cache_key] = (_emb_vis, _emb_hid)
                    print(f"  init={_igs}, {_crit}: computed ({len(_emb_hid)} hidden, {time.time()-_t0:.1f}s)")
                else:
                    print(f"  init={_igs}, {_crit}: embedding FAILED — will skip this config")

print(f"\nArchitecture assignments cached:")
print(f"  Tiling: 3 configs (patch_size=2,4,7)")
print(f"  ILP: 1 canonical config")
print(f"  Embedding: {sum(1 for k in _arch_assignment_cache if k.startswith('embedding_'))} configs")
print(f"  Spectral: computed per trial (random refinement)")


# =============================================================================
# Leaderboard Management
# =============================================================================

def load_leaderboard(path: str) -> list:
    """Load the leaderboard from JSON, or return empty list."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_leaderboard(leaderboard: list, path: str):
    """Save the leaderboard to JSON (atomic write via temp file)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(leaderboard, f, indent=2, default=str)
    os.replace(tmp_path, path)


def update_leaderboard(leaderboard: list, entry: dict, top_n: int = TOP_N) -> list:
    """
    Insert a new entry into the leaderboard, keeping only the top-N by best FID.
    Returns the (possibly trimmed) leaderboard, sorted by best_fid ascending.
    """
    leaderboard.append(entry)
    leaderboard.sort(key=lambda e: e["best_fid"])
    
    # If we need to trim, remove excess entries and their model files
    if len(leaderboard) > top_n:
        for removed in leaderboard[top_n:]:
            model_path = removed.get("model_path", "")
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
                print(f"  Removed outranked model: {os.path.basename(model_path)}")
        leaderboard = leaderboard[:top_n]
    
    return leaderboard


# =============================================================================
# Architecture Construction
# =============================================================================

def build_architecture(config: dict, G: nx.Graph):
    """
    Build the architecture (V/H partition + pixel assignment) for a given config.

    Returns:
        (G_relabeled, node_labels, num_hidden, arch_stats, arch_time)
    """
    t0 = time.time()
    arch = config["architecture"]
    num_visible = config["num_visible"]
    grid_shape = tuple(config["grid_shape"])

    if arch == "spectral":
        visible_set = spectral_bisection_partition(G, num_visible)
        visible_set = refine_partition_swaps(
            G, visible_set, num_visible,
            max_iters=config["refinement_iters"],
            gamma=config["vv_penalty"],
        )
        visible_in_pixel_order = assign_pixel_order_spatial_arch1(G, visible_set, grid_shape)
        hidden_nodes = [n for n in sorted(G.nodes()) if n not in visible_set]

    elif arch == "tiling":
        cache_key = f"tiling_patch{config['patch_size']}"
        visible_in_pixel_order, hidden_nodes = _arch_assignment_cache[cache_key]

    elif arch == "ilp":
        visible_in_pixel_order, hidden_nodes = _arch_assignment_cache["ilp_canonical"]

    elif arch == "embedding":
        cache_key = f"embedding_init{config['initial_grid_size']}_{config['add_node_criterion']}"
        if cache_key not in _arch_assignment_cache:
            # Not pre-cached (embedding failed during pre-computation)
            raise RuntimeError(
                f"Embedding assignment not available for init={config['initial_grid_size']}, "
                f"criterion={config['add_node_criterion']}. Embedding likely failed for this config."
            )
        visible_in_pixel_order, hidden_nodes = _arch_assignment_cache[cache_key]

    num_hidden = len(hidden_nodes)

    G_relabeled, mapping = relabel_visible_first(G, visible_in_pixel_order)
    node_labels = {}
    for i in range(num_visible):
        node_labels[i] = "visible"
    for i in range(num_visible, num_visible + num_hidden):
        node_labels[i] = "hidden"

    # Compute architecture stats
    visible_relabeled = list(range(num_visible))
    hidden_relabeled = list(range(num_visible, num_visible + num_hidden))

    vh_edges = sum(
        1 for u, v in G_relabeled.edges()
        if (u < num_visible) != (v < num_visible)
    )
    vv_edges = sum(
        1 for u, v in G_relabeled.edges()
        if u < num_visible and v < num_visible
    )
    hh_edges = sum(
        1 for u, v in G_relabeled.edges()
        if u >= num_visible and v >= num_visible
    )
    arch_stats = {"vh": vh_edges, "vv": vv_edges, "hh": hh_edges}

    arch_time = time.time() - t0
    return G_relabeled, node_labels, num_hidden, arch_stats, arch_time


# =============================================================================
# Single Trial
# =============================================================================

def run_trial(
    trial_num: int,
    config: dict,
    G_zephyr: nx.Graph,
    train_data: torch.Tensor,
    loader: DataLoader,
    classifier: MNISTQualityClassifier,
    real_features_cache: np.ndarray,
) -> dict:
    """
    Run a single hyperparameter trial.

    Returns a dict with results (or None if the trial failed).
    """
    arch = config["architecture"]
    print(f"\n{'='*70}")
    print(f"  TRIAL {trial_num} — {arch.upper()}")
    print(f"  lr={config['lr']:.2e}, wd={config['weight_decay']:.2e}, "
          f"bs={config['batch_size']}, ep={config['epochs']}, k={config['k_steps']}, "
          f"PCD={config['persistent']}")
    if arch == "spectral":
        print(f"  refine={config['refinement_iters']}, vv_pen={config['vv_penalty']:.4f}")
    elif arch == "tiling":
        print(f"  patch_size={config['patch_size']}")
    elif arch == "ilp":
        print(f"  (using pre-solved canonical assignment)")
    elif arch == "embedding":
        print(f"  init_grid={config['initial_grid_size']}, criterion={config['add_node_criterion']}")
    print(f"{'='*70}")

    trial_start = time.time()

    # --- 1. Build architecture ---
    try:
        print("\n[1/4] Building architecture...")
        G_relabeled, node_labels, num_hidden, arch_stats, arch_time = build_architecture(
            config, G_zephyr
        )
        config["num_hidden"] = num_hidden
        print(f"  Hidden: {num_hidden}, VH edges: {arch_stats['vh']}, "
              f"VV: {arch_stats['vv']}, HH: {arch_stats['hh']}  ({arch_time:.1f}s)")
    except Exception as e:
        print(f"  Architecture construction FAILED: {e}")
        return None

    # --- 2. Train ---
    try:
        print(f"\n[2/4] Training ({config['epochs']} epochs, k={config['k_steps']})...")
        model = graph_to_bm(G_relabeled, node_labels)
        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        training_history = train_boltzmann_machine_pcd(
            model, loader, optimizer,
            num_epochs=config["epochs"],
            k_steps=config["k_steps"],
            batch_size=config["batch_size"],
            step_size=config["lr"],
            persistent=config["persistent"],
            train_data=train_data,
            eval_every=max(1, config["epochs"] // 5),  # ~5 eval points
        )
        train_time = time.time() - trial_start - arch_time
        print(f"  Training time: {train_time:.1f}s")
    except Exception as e:
        print(f"  Training FAILED: {e}")
        import traceback; traceback.print_exc()
        return None

    # --- 3. Evaluate sample quality ---
    try:
        print(f"\n[3/4] Evaluating sample quality ({EVAL_NUM_SAMPLES} samples)...")
        quality_report = eval_samples_fullMNIST(
            model,
            real_data=train_data,
            grid_shape=tuple(config["grid_shape"]),
            num_samples=EVAL_NUM_SAMPLES,
            classifier=classifier,
            gibbs_burn_in=2000,
            batched_sa_start_temp=10.0,
            batched_sa_end_temp=0.1,
            batched_sa_iterations=20,
            sampling_methods=EVAL_METHODS,
            n_fid_bootstrap=100,
            real_features_cache=real_features_cache,
            verbose=True,
        )
        eval_time = time.time() - trial_start - arch_time - train_time
    except Exception as e:
        print(f"  Evaluation FAILED: {e}")
        import traceback; traceback.print_exc()
        return None

    # --- 4. Save model ---
    best_fid = quality_report["best_fid"]
    best_method = quality_report["best_method"]
    total_time = time.time() - trial_start

    # Build a compact tag
    method_tag = "PCD" if config["persistent"] else "CD"
    model_filename = (
        f"hpsearch_{RUN_TIMESTAMP}_trial{trial_num:03d}_{arch}_"
        f"{method_tag}_lr{config['lr']:.1e}_ep{config['epochs']}_"
        f"fid{best_fid:.0f}_model.pt"
    )
    model_path = os.path.join(DATA_DIR, model_filename)

    print(f"\n[4/4] Saving model...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "training_history": training_history,
        "arch_name": arch,
        "hyperparams": config,
        "graph_edges": list(G_relabeled.edges()),
        "node_labels": node_labels,
        "quality_report": {
            "best_method": best_method,
            "best_fid": best_fid,
            "methods": {
                m: {
                    "fid": r["fid"],
                    "classifier": {
                        k: v for k, v in r["classifier"].items()
                        if k != "predicted_classes"
                    },
                    "time_seconds": r["time_seconds"],
                }
                for m, r in quality_report["sampling_results"].items()
            },
        },
    }, model_path)

    # Save a sample grid for visual inspection
    for method_name, method_data in quality_report["sampling_results"].items():
        samples_np = method_data["samples"].numpy()
        n_show = min(16, len(samples_np))
        nr = int(np.ceil(np.sqrt(n_show)))
        nc = int(np.ceil(n_show / nr))
        fig, axes = plt.subplots(nr, nc, figsize=(2 * nc, 2 * nr))
        fid_val = method_data["fid"]["fid"]
        fig.suptitle(
            f"Trial {trial_num} {arch} {method_name} (FID={fid_val:.1f})"
        )
        for i, ax in enumerate(axes.flat):
            if i < n_show:
                ax.imshow(
                    samples_np[i].reshape(tuple(config["grid_shape"])),
                    cmap="gray", vmin=0, vmax=1,
                )
            ax.axis("off")
        plt.tight_layout()
        fig_path = os.path.join(
            DATA_DIR,
            f"hpsearch_{RUN_TIMESTAMP}_trial{trial_num:03d}_{arch}_{method_name}.png",
        )
        plt.savefig(fig_path, dpi=100)
        plt.close(fig)

    # Build result entry
    result = {
        "trial": trial_num,
        "timestamp": datetime.now().isoformat(),
        "best_fid": best_fid,
        "best_method": best_method,
        "model_path": model_path,
        "config": config,
        "arch_stats": arch_stats,
        "total_time_seconds": total_time,
        "fid_per_method": {
            m: {
                "fid": r["fid"]["fid"],
                "ci_low": r["fid"]["ci_low"],
                "ci_high": r["fid"]["ci_high"],
                "fid_std": r["fid"]["fid_std"],
                "mean_confidence": r["classifier"]["mean_confidence"],
                "frac_high_confidence": r["classifier"]["frac_high_confidence"],
                "class_balance_entropy": r["classifier"]["class_balance_entropy"],
            }
            for m, r in quality_report["sampling_results"].items()
        },
        "final_pll": training_history["pll"][-1] if training_history["pll"] else None,
        "final_pcd_loss": training_history["pcd_loss"][-1] if training_history["pcd_loss"] else None,
    }

    print(f"\n  ✓ Trial {trial_num} complete in {total_time:.1f}s")
    print(f"    Best FID: {best_fid:.2f} ({best_method})")
    for m, r in quality_report["sampling_results"].items():
        fid_info = r["fid"]
        print(f"    {m}: FID={fid_info['fid']:.2f} "
              f"[{fid_info['ci_low']:.2f}, {fid_info['ci_high']:.2f}]  "
              f"conf={r['classifier']['mean_confidence']:.3f}")

    # Free GPU memory
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# =============================================================================
# Main Search Loop
# =============================================================================

def main():
    # Resume from existing leaderboard if provided
    if args.resume and os.path.exists(args.resume):
        leaderboard = load_leaderboard(args.resume)
        print(f"\nResumed leaderboard from {args.resume} ({len(leaderboard)} entries)")
        # Use the resume path as the leaderboard path
        leaderboard_path = args.resume
    else:
        leaderboard = []
        leaderboard_path = LEADERBOARD_PATH

    search_start = time.time()
    trial_num = 0
    completed_trials = 0
    failed_trials = 0

    print(f"\n{'#'*70}")
    print(f"  SEARCH STARTING — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Budget: {args.hours}h = {TIME_BUDGET_SECONDS:.0f}s")
    print(f"{'#'*70}\n")

    while True:
        elapsed = time.time() - search_start
        remaining = TIME_BUDGET_SECONDS - elapsed

        # Check stopping conditions
        if elapsed >= TIME_BUDGET_SECONDS:
            print(f"\n⏰ Time budget exhausted ({args.hours}h)")
            break
        if completed_trials + failed_trials >= MAX_TRIALS:
            print(f"\n🏁 Max trials reached ({MAX_TRIALS})")
            break

        trial_num += 1
        print(f"\n{'─'*70}")
        print(f"  Elapsed: {elapsed/3600:.2f}h / {args.hours}h | "
              f"Remaining: {remaining/3600:.2f}h | "
              f"Trials: {completed_trials} done, {failed_trials} failed")
        if leaderboard:
            print(f"  Current best FID: {leaderboard[0]['best_fid']:.2f} "
                  f"(trial {leaderboard[0]['trial']}, {leaderboard[0]['config']['architecture']})")
        print(f"{'─'*70}")

        # Sample config
        config = sample_hyperparams()

        # Estimate if we have enough time (rough: ~2 min per epoch + eval)
        est_minutes = config["epochs"] * 2 + 10  # rough estimate
        if remaining < est_minutes * 60 * 0.5:
            # Not enough time, try a shorter config
            config["epochs"] = min(config["epochs"], max(15, int(remaining / 180)))
            print(f"  (Reduced epochs to {config['epochs']} due to time remaining)")

        # Build loader with this trial's batch size
        loader = DataLoader(
            TensorDataset(train_data),
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        # Run trial
        result = run_trial(
            trial_num, config, G_zephyr,
            train_data, loader,
            classifier, real_features_cache,
        )

        if result is None:
            failed_trials += 1
            continue

        completed_trials += 1

        # Update leaderboard
        leaderboard = update_leaderboard(leaderboard, result, top_n=TOP_N)
        save_leaderboard(leaderboard, leaderboard_path)
        print(f"\n  📊 Leaderboard updated ({leaderboard_path})")

        # Print current leaderboard
        print(f"\n  {'='*60}")
        print(f"  TOP-{TOP_N} LEADERBOARD (after {completed_trials} trials)")
        print(f"  {'='*60}")
        for rank, entry in enumerate(leaderboard, 1):
            fid = entry["best_fid"]
            arch = entry["config"]["architecture"]
            lr = entry["config"]["lr"]
            ep = entry["config"]["epochs"]
            meth = entry["best_method"]
            t = entry["trial"]
            marker = " ← NEW" if t == trial_num else ""
            print(f"  {rank}. Trial {t:3d}  FID={fid:7.2f}  {arch:10s}  "
                  f"lr={lr:.1e}  ep={ep}  ({meth}){marker}")
        print()

    # Final summary
    total_time = time.time() - search_start
    print(f"\n{'#'*70}")
    print(f"  SEARCH COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time: {total_time/3600:.2f}h")
    print(f"  Trials: {completed_trials} completed, {failed_trials} failed")
    print(f"  Leaderboard: {leaderboard_path}")
    print(f"{'#'*70}")

    if leaderboard:
        print(f"\n  FINAL TOP-{TOP_N}:")
        print(f"  {'─'*60}")
        for rank, entry in enumerate(leaderboard, 1):
            fid = entry["best_fid"]
            arch = entry["config"]["architecture"]
            lr = entry["config"]["lr"]
            wd = entry["config"]["weight_decay"]
            ep = entry["config"]["epochs"]
            k = entry["config"]["k_steps"]
            bs = entry["config"]["batch_size"]
            meth = entry["best_method"]
            t = entry["trial"]
            print(f"\n  #{rank} — Trial {t}, FID={fid:.2f} ({meth})")
            print(f"    arch={arch}, lr={lr:.2e}, wd={wd:.2e}, "
                  f"bs={bs}, ep={ep}, k={k}")
            if arch == "spectral":
                print(f"    refine={entry['config']['refinement_iters']}, "
                      f"vv_pen={entry['config']['vv_penalty']:.4f}")
            elif arch == "tiling":
                print(f"    patch={entry['config']['patch_size']}")
            elif arch == "ilp":
                print(f"    (pre-solved canonical assignment)")
            elif arch == "embedding":
                print(f"    init_grid={entry['config']['initial_grid_size']}, "
                      f"criterion={entry['config']['add_node_criterion']}")
            print(f"    Model: {entry['model_path']}")

    print(f"\nDone. Results saved to {leaderboard_path}")


if __name__ == "__main__":
    main()
