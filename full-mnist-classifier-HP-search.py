"""
full-mnist-classifier-HP-search.py

Random hyperparameter search for the discriminative BM classifier used in
full-mnist-classifier-test.py.

This script keeps the hardware graph fixed to Zephyr Z(12), builds a native
Zephyr classifier architecture for each trial, trains on a train split of full
MNIST, selects the best checkpoint by validation accuracy with early stopping,
and then reports test accuracy for that best validation checkpoint.

The classifier uses only nodes and edges already present in the Zephyr graph.
For each trial, it first builds the base spectral or tiling architecture, then
selects label nodes from the existing hidden-node pool using a native structural
coverage objective.

Basic usage:
	python full-mnist-classifier-HP-search.py

Run for a fixed budget and trial count:
	python full-mnist-classifier-HP-search.py --hours 12 --max-trials 40

Search only over spectral architectures:
	python full-mnist-classifier-HP-search.py --architectures spectral

Resume from an existing leaderboard:
	python full-mnist-classifier-HP-search.py --resume data/EXISTING_RESULTS.json

Restrict to specific digits, for example 0/1/7:
	python full-mnist-classifier-HP-search.py --digit-filter 0 1 7

Example focused run:
	python full-mnist-classifier-HP-search.py \
		--hours 8 \
		--max-trials 30 \
		--architectures spectral \
		--patience 8 \
		--min-epochs 8 \
		--seed 42 \
		--verbose-training

Arguments:
	--hours
		Time budget in hours for the full search loop.

	--max-trials
		Maximum number of trials to run before stopping.

	--top-n
		How many top checkpoints to keep on disk and in the leaderboard.

	--seed
		Random seed for reproducible hyperparameter sampling and data splitting.

	--resume
		Path to an existing leaderboard JSON file to continue updating.

	--architectures
		Which architecture families to sample from. Current options: spectral, tiling.

	--digit-filter
		Optional subset of MNIST digits to use instead of all 10 classes.

	--validation-fraction
		Fraction of the MNIST training set reserved for validation.

	--patience
		Early-stopping patience in epochs.

	--min-epochs
		Minimum number of epochs before early stopping is allowed to trigger.

	--eval-batch-size
		Batch size for validation and test evaluation.

	--min-delta
		Minimum validation-accuracy improvement needed to count as progress.

	--verbose-training
		Print per-epoch training and validation metrics.

	--label-coverage-mode
		Objective used when assigning native label nodes.
		'hidden_only' scores coverage only over remaining hidden nodes.
		'hidden_visible' scores coverage over both hidden and visible nodes.

	--label-assignment-time-limit
		Maximum time in seconds spent improving the native label-node assignment
		for each architecture build.

Outputs:
	- A leaderboard JSON in data/ updated after every completed trial.
	- Model checkpoints for the current top-N trials.
	- Per-trial metadata including sampled hyperparameters, native label-node
	  assignment diagnostics, validation accuracy, and test accuracy.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from datetime import datetime

import dwave_networkx as dnx
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from bolmaqua import device, graph_to_bm, relabel_visible_first
from classifierhelper import (
	build_native_labelled_graph,
	compute_class_scores_free_energy,
	prepare_classification_batch,
)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module


_arch1 = import_module("ARCH1-spectral-bisection")
spectral_bisection_partition = _arch1.spectral_bisection_partition
refine_partition_swaps = _arch1.refine_partition_swaps
assign_pixel_order_spatial_arch1 = _arch1.assign_pixel_order_spatial

_arch2 = import_module("ARCH2-hierarchical-tiling")
tiling_assignment = _arch2.tiling_assignment


VALID_ARCHITECTURES = ("spectral", "tiling")


parser = argparse.ArgumentParser(
	description="Hyperparameter search for full-MNIST BM classification on Zephyr Z(12)"
)
parser.add_argument("--hours", type=float, default=8.0, help="Time budget in hours")
parser.add_argument("--max-trials", type=int, default=999, help="Maximum number of trials")
parser.add_argument("--top-n", type=int, default=5, help="Number of best runs to retain")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument(
	"--resume",
	type=str,
	default=None,
	help="Path to an existing leaderboard JSON to resume from",
)
parser.add_argument(
	"--architectures",
	nargs="+",
	choices=VALID_ARCHITECTURES,
	default=list(VALID_ARCHITECTURES),
	help="Architectures to include in the search pool",
)
parser.add_argument(
	"--digit-filter",
	nargs="+",
	type=int,
	default=None,
	help="Optional subset of MNIST digits to use, e.g. --digit-filter 0 1 7",
)
parser.add_argument(
	"--validation-fraction",
	type=float,
	default=0.1,
	help="Fraction of the training split reserved for validation",
)
parser.add_argument(
	"--patience",
	type=int,
	default=6,
	help="Early stopping patience in epochs",
)
parser.add_argument(
	"--min-epochs",
	type=int,
	default=5,
	help="Minimum epochs before early stopping can trigger",
)
parser.add_argument(
	"--eval-batch-size",
	type=int,
	default=512,
	help="Batch size for validation/test evaluation",
)
parser.add_argument(
	"--min-delta",
	type=float,
	default=1e-4,
	help="Minimum validation accuracy improvement required to reset patience",
)
parser.add_argument(
	"--verbose-training",
	action="store_true",
	help="Print per-epoch training and validation metrics",
)
parser.add_argument(
	"--label-coverage-mode",
	choices=["hidden_only", "hidden_visible"],
	default="hidden_visible",
	help="Which nodes count toward native label-node coverage during assignment",
)
parser.add_argument(
	"--label-assignment-time-limit",
	type=float,
	default=120.0,
	help="Time limit in seconds for improving the native label-node assignment",
)
args = parser.parse_args()


GRID_SHAPE = (28, 28)
NUM_VISIBLE_PIXELS = GRID_SHAPE[0] * GRID_SHAPE[1]
BINARIZE_THRESHOLD = 0.5
K = 12
N_ZEPHYR = 16 * K * (2 * K + 1)
DATA_DIR = "data"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LEADERBOARD_PATH = os.path.join(
	DATA_DIR, f"{RUN_TIMESTAMP}_classifier_hyperparam_best_results.json"
)
TIME_BUDGET_SECONDS = args.hours * 3600
MAX_TRIALS = args.max_trials
TOP_N = args.top_n


if args.validation_fraction <= 0.0 or args.validation_fraction >= 0.5:
	raise ValueError("validation_fraction must be in (0.0, 0.5)")

if args.min_epochs < 1:
	raise ValueError("min_epochs must be at least 1")

if args.patience < 1:
	raise ValueError("patience must be at least 1")

if N_ZEPHYR < NUM_VISIBLE_PIXELS:
	raise ValueError(
		f"Zephyr Z({K}) has only {N_ZEPHYR} nodes, need at least {NUM_VISIBLE_PIXELS}."
	)

os.makedirs(DATA_DIR, exist_ok=True)

if args.seed is not None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)


def sample_hyperparams() -> dict:
	architecture = random.choice(args.architectures)
	config = {
		"architecture": architecture,
		"K": K,
		"grid_shape": list(GRID_SHAPE),
		"num_visible_pixels": NUM_VISIBLE_PIXELS,
		"binarize_threshold": BINARIZE_THRESHOLD,
		"digit_filter": sorted(args.digit_filter) if args.digit_filter is not None else None,
		"lr": 10 ** random.uniform(np.log10(5e-4), np.log10(2e-3)),
		"weight_decay": 10 ** random.uniform(-8.0, np.log10(3e-5)),
		"batch_size": random.choice([32, 64]),
		"epochs": random.choice([20, 25, 30, 35]),
		"k_steps": random.choice([3, 5, 10]),
		"nodes_per_label": 9,
		"classification_loss_weight": 10 ** random.uniform(np.log10(0.5), np.log10(2.0)),
		"classification_inference_method": "free_energy",
		"label_assignment_mode": "native_coverage",
		"label_coverage_mode": args.label_coverage_mode,
		"label_assignment_time_limit": args.label_assignment_time_limit,
	}

	if architecture == "spectral":
		config["refinement_iters"] = random.choice([500, 1000, 1500])
		config["vv_penalty"] = 10 ** random.uniform(-3.0, np.log10(4e-3))
	elif architecture == "tiling":
		config["patch_size"] = random.choice([4, 7])

	return config


def prepare_mnist(dataset, digit_filter=None, threshold=0.5, label_mapping=None):
	images = dataset.data.float() / 255.0
	labels = dataset.targets.clone()

	if digit_filter is not None:
		mask = torch.zeros(len(labels), dtype=torch.bool)
		for digit in digit_filter:
			mask |= labels == digit
		images = images[mask]
		labels = labels[mask]
		if label_mapping is not None:
			labels = torch.tensor([label_mapping[int(label)] for label in labels], dtype=torch.long)

	images = (images >= threshold).float()
	images = images.view(images.size(0), -1)
	return images, labels.long()


def stratified_split_indices(labels: torch.Tensor, val_fraction: float, seed: int | None):
	rng = np.random.default_rng(seed)
	labels_np = labels.cpu().numpy()
	train_indices = []
	val_indices = []

	for class_id in np.unique(labels_np):
		class_indices = np.where(labels_np == class_id)[0]
		rng.shuffle(class_indices)
		if len(class_indices) == 1:
			train_indices.extend(class_indices.tolist())
			continue

		n_val = int(round(len(class_indices) * val_fraction))
		n_val = max(1, min(len(class_indices) - 1, n_val))

		val_indices.extend(class_indices[:n_val].tolist())
		train_indices.extend(class_indices[n_val:].tolist())

	rng.shuffle(train_indices)
	rng.shuffle(val_indices)
	return torch.tensor(train_indices, dtype=torch.long), torch.tensor(val_indices, dtype=torch.long)


def build_data_splits():
	digit_filter = sorted(args.digit_filter) if args.digit_filter is not None else None
	label_mapping = None
	if digit_filter is None:
		num_classes = 10
	else:
		num_classes = len(digit_filter)
		label_mapping = {digit: index for index, digit in enumerate(digit_filter)}

	print("=" * 70)
	print("  FULL MNIST CLASSIFIER HYPERPARAMETER SEARCH")
	print(f"  Fixed Zephyr graph: Z({K}) with {N_ZEPHYR} nodes")
	print(f"  Time budget: {args.hours}h | Max trials: {MAX_TRIALS} | Top-{TOP_N} kept")
	print(f"  Validation fraction: {args.validation_fraction:.2f} | Patience: {args.patience}")
	print(f"  Leaderboard: {LEADERBOARD_PATH if args.resume is None else args.resume}")
	print("=" * 70)

	print("\nLoading full MNIST dataset...")
	train_dataset = datasets.MNIST(
		root="./data", train=True, download=True, transform=transforms.ToTensor()
	)
	test_dataset = datasets.MNIST(
		root="./data", train=False, download=True, transform=transforms.ToTensor()
	)

	train_images_all, train_labels_all = prepare_mnist(
		train_dataset, digit_filter, BINARIZE_THRESHOLD, label_mapping
	)
	test_images, test_labels = prepare_mnist(
		test_dataset, digit_filter, BINARIZE_THRESHOLD, label_mapping
	)

	split_seed = args.seed if args.seed is not None else 0
	train_indices, val_indices = stratified_split_indices(
		train_labels_all, args.validation_fraction, split_seed
	)

	train_images = train_images_all[train_indices]
	train_labels = train_labels_all[train_indices]
	val_images = train_images_all[val_indices]
	val_labels = train_labels_all[val_indices]

	print(f"  Train: {len(train_images)} samples")
	print(f"  Val:   {len(val_images)} samples")
	print(f"  Test:  {len(test_images)} samples")
	print(f"  Classes: {num_classes}")

	return {
		"train_images": train_images,
		"train_labels": train_labels,
		"val_images": val_images,
		"val_labels": val_labels,
		"test_images": test_images,
		"test_labels": test_labels,
		"num_classes": num_classes,
		"digit_filter": digit_filter,
	}


print(f"\nGenerating Zephyr graph Z({K})...")
G_ZEPHYR = dnx.zephyr_graph(K)
print(f"  Nodes: {G_ZEPHYR.number_of_nodes()}, Edges: {G_ZEPHYR.number_of_edges()}")


ARCH_ASSIGNMENT_CACHE = {}


def precompute_architecture_assignments():
	if "tiling" in args.architectures:
		print("\nPre-computing tiling assignments...")
		for patch_size in [2, 4, 7]:
			t0 = time.time()
			visible_order, hidden_nodes = tiling_assignment(G_ZEPHYR, GRID_SHAPE, patch_size)
			ARCH_ASSIGNMENT_CACHE[f"tiling_patch{patch_size}"] = (visible_order, hidden_nodes)
			print(
				f"  patch_size={patch_size}: {len(hidden_nodes)} hidden nodes "
				f"({time.time() - t0:.1f}s)"
			)


def load_leaderboard(path: str) -> list:
	if os.path.exists(path):
		with open(path, "r") as handle:
			return json.load(handle)
	return []


def save_leaderboard(leaderboard: list, path: str):
	temp_path = path + ".tmp"
	with open(temp_path, "w") as handle:
		json.dump(leaderboard, handle, indent=2, default=str)
	os.replace(temp_path, path)


def leaderboard_sort_key(entry: dict):
	return (-entry["best_val_acc"], entry["best_val_loss"], -entry["test_acc"])


def update_leaderboard(leaderboard: list, entry: dict, top_n: int) -> list:
	leaderboard.append(entry)
	leaderboard.sort(key=leaderboard_sort_key)

	if len(leaderboard) > top_n:
		for removed in leaderboard[top_n:]:
			model_path = removed.get("model_path")
			if model_path and os.path.exists(model_path):
				os.remove(model_path)
				print(f"  Removed outranked model: {os.path.basename(model_path)}")
		leaderboard = leaderboard[:top_n]

	return leaderboard


def build_architecture(config: dict, num_classes: int):
	t0 = time.time()
	architecture = config["architecture"]

	if architecture == "spectral":
		visible_set = spectral_bisection_partition(G_ZEPHYR, NUM_VISIBLE_PIXELS)
		visible_set = refine_partition_swaps(
			G_ZEPHYR,
			visible_set,
			NUM_VISIBLE_PIXELS,
			max_iters=config["refinement_iters"],
			gamma=config["vv_penalty"],
		)
		visible_in_pixel_order = assign_pixel_order_spatial_arch1(
			G_ZEPHYR, visible_set, GRID_SHAPE
		)
		hidden_nodes = [node for node in sorted(G_ZEPHYR.nodes()) if node not in visible_set]
	elif architecture == "tiling":
		visible_in_pixel_order, hidden_nodes = ARCH_ASSIGNMENT_CACHE[
			f"tiling_patch{config['patch_size']}"
		]
	else:
		raise ValueError(f"Unknown architecture: {architecture}")

	num_hidden = len(hidden_nodes)
	graph_relabeled, _ = relabel_visible_first(G_ZEPHYR, visible_in_pixel_order)

	pixel_visible_nodes = list(range(NUM_VISIBLE_PIXELS))
	hidden_relabeled = list(range(NUM_VISIBLE_PIXELS, NUM_VISIBLE_PIXELS + num_hidden))

	vh_edges = sum(
		1 for u, v in graph_relabeled.edges() if (u < NUM_VISIBLE_PIXELS) != (v < NUM_VISIBLE_PIXELS)
	)
	vv_edges = sum(
		1 for u, v in graph_relabeled.edges() if u < NUM_VISIBLE_PIXELS and v < NUM_VISIBLE_PIXELS
	)
	hh_edges = sum(
		1
		for u, v in graph_relabeled.edges()
		if u >= NUM_VISIBLE_PIXELS and v >= NUM_VISIBLE_PIXELS
	)

	graph_extended, label_node_groups, remaining_hidden_nodes, node_labels, label_assignment_stats, _ = (
		build_native_labelled_graph(
			graph_relabeled,
			pixel_visible_nodes,
			hidden_relabeled,
			num_classes=num_classes,
			nodes_per_label=config["nodes_per_label"],
			coverage_mode=config["label_coverage_mode"],
			time_limit_seconds=config["label_assignment_time_limit"],
			seed=args.seed,
			verbose=True,
		)
	)

	arch_stats = {
		"vh": vh_edges,
		"vv": vv_edges,
		"hh": hh_edges,
		"num_hidden_before_labels": num_hidden,
		"num_hidden": len(remaining_hidden_nodes),
		"num_label_nodes": num_classes * config["nodes_per_label"],
		"label_assignment": label_assignment_stats,
	}

	return graph_extended, node_labels, label_node_groups, arch_stats, time.time() - t0


def make_loader(images, labels, batch_size, shuffle, drop_last):
	return DataLoader(
		TensorDataset(images, labels),
		batch_size=batch_size,
		shuffle=shuffle,
		drop_last=drop_last,
	)


def evaluate_classification_model(model, loader, label_node_groups, num_classes, nodes_per_label):
	was_training = model.training
	model.eval()

	total_loss = 0.0
	total_correct = 0
	total_examples = 0
	per_class_correct = {class_id: 0 for class_id in range(num_classes)}
	per_class_total = {class_id: 0 for class_id in range(num_classes)}

	with torch.inference_mode():
		for pixels, labels in loader:
			pixels = pixels.to(device)
			labels = labels.to(device)

			class_scores = compute_class_scores_free_energy(
				model,
				pixels,
				label_node_groups,
				num_classes=num_classes,
				nodes_per_label=nodes_per_label,
			)
			cls_loss = F.cross_entropy(class_scores, labels)
			preds = class_scores.argmax(dim=1)

			batch_size = labels.size(0)
			total_loss += cls_loss.item() * batch_size
			total_correct += (preds == labels).sum().item()
			total_examples += batch_size

			preds_cpu = preds.cpu()
			labels_cpu = labels.cpu()
			for class_id in range(num_classes):
				class_mask = labels_cpu == class_id
				class_count = int(class_mask.sum().item())
				if class_count == 0:
					continue
				per_class_total[class_id] += class_count
				per_class_correct[class_id] += int((preds_cpu[class_mask] == labels_cpu[class_mask]).sum().item())

	if was_training:
		model.train()

	average_loss = total_loss / max(1, total_examples)
	accuracy = total_correct / max(1, total_examples)
	per_class_acc = {
		class_id: (per_class_correct[class_id] / per_class_total[class_id])
		if per_class_total[class_id] > 0
		else 0.0
		for class_id in range(num_classes)
	}
	return average_loss, accuracy, per_class_acc


def train_with_early_stopping(
	model,
	train_loader,
	val_loader,
	config,
	label_node_groups,
	num_classes,
):
	optimizer = torch.optim.Adam(
		model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
	)

	history = {
		"train_cd_loss": [],
		"train_cls_loss": [],
		"train_total_loss": [],
		"train_acc": [],
		"val_cls_loss": [],
		"val_acc": [],
		"best_epoch": 0,
		"stopped_early": False,
	}

	best_state = None
	best_val_acc = -1.0
	best_val_loss = float("inf")
	patience_counter = 0

	for epoch in range(config["epochs"]):
		model.train()
		epoch_cd_loss = 0.0
		epoch_cls_loss = 0.0
		epoch_total_loss = 0.0
		epoch_correct = 0
		epoch_examples = 0

		for pixels, labels in train_loader:
			pixels = pixels.to(device)
			labels = labels.to(device)

			extended_visible = prepare_classification_batch(
				pixels,
				labels,
				label_node_groups,
				num_classes=num_classes,
				nodes_per_label=config["nodes_per_label"],
			)

			optimizer.zero_grad(set_to_none=True)
			cd_loss, _ = model(extended_visible, k_steps=config["k_steps"])
			class_scores = compute_class_scores_free_energy(
				model,
				pixels,
				label_node_groups,
				num_classes=num_classes,
				nodes_per_label=config["nodes_per_label"],
			)
			cls_loss = F.cross_entropy(class_scores, labels)
			total_loss = cd_loss + config["classification_loss_weight"] * cls_loss
			total_loss.backward()
			optimizer.step()

			batch_size = labels.size(0)
			epoch_cd_loss += cd_loss.item() * batch_size
			epoch_cls_loss += cls_loss.item() * batch_size
			epoch_total_loss += total_loss.item() * batch_size
			epoch_correct += (class_scores.argmax(dim=1) == labels).sum().item()
			epoch_examples += batch_size

		train_cd_loss = epoch_cd_loss / max(1, epoch_examples)
		train_cls_loss = epoch_cls_loss / max(1, epoch_examples)
		train_total_loss = epoch_total_loss / max(1, epoch_examples)
		train_acc = epoch_correct / max(1, epoch_examples)

		val_cls_loss, val_acc, _ = evaluate_classification_model(
			model,
			val_loader,
			label_node_groups,
			num_classes,
			config["nodes_per_label"],
		)

		history["train_cd_loss"].append(train_cd_loss)
		history["train_cls_loss"].append(train_cls_loss)
		history["train_total_loss"].append(train_total_loss)
		history["train_acc"].append(train_acc)
		history["val_cls_loss"].append(val_cls_loss)
		history["val_acc"].append(val_acc)

		if args.verbose_training:
			print(
				f"    Epoch {epoch + 1}/{config['epochs']} | "
				f"train_total={train_total_loss:.4f} | train_acc={train_acc:.4f} | "
				f"val_cls={val_cls_loss:.4f} | val_acc={val_acc:.4f}"
			)

		improved = False
		if val_acc > best_val_acc + args.min_delta:
			improved = True
		elif abs(val_acc - best_val_acc) <= args.min_delta and val_cls_loss < best_val_loss:
			improved = True

		if improved:
			best_val_acc = val_acc
			best_val_loss = val_cls_loss
			best_state = copy.deepcopy(model.state_dict())
			history["best_epoch"] = epoch + 1
			patience_counter = 0
		elif epoch + 1 >= args.min_epochs:
			patience_counter += 1
			if patience_counter >= args.patience:
				history["stopped_early"] = True
				if args.verbose_training:
					print(
						f"    Early stopping at epoch {epoch + 1} "
						f"(best val_acc={best_val_acc:.4f} at epoch {history['best_epoch']})"
					)
				break

	if best_state is not None:
		model.load_state_dict(best_state)

	return model, history, best_val_acc, best_val_loss


def run_trial(trial_num: int, config: dict, data_splits: dict) -> dict | None:
	print(f"\n{'=' * 70}")
	print(f"  TRIAL {trial_num} — {config['architecture'].upper()}")
	print(
		f"  lr={config['lr']:.2e}, wd={config['weight_decay']:.2e}, "
		f"bs={config['batch_size']}, ep={config['epochs']}, k={config['k_steps']}, "
		f"nodes/label={config['nodes_per_label']}, cls_w={config['classification_loss_weight']:.3f}"
	)
	if config["architecture"] == "spectral":
		print(
			f"  refine={config['refinement_iters']}, vv_penalty={config['vv_penalty']:.4f}"
		)
	elif config["architecture"] == "tiling":
		print(f"  patch_size={config['patch_size']}")
	print(f"{'=' * 70}")

	trial_start = time.time()
	train_loader = make_loader(
		data_splits["train_images"],
		data_splits["train_labels"],
		config["batch_size"],
		shuffle=True,
		drop_last=True,
	)
	val_loader = make_loader(
		data_splits["val_images"],
		data_splits["val_labels"],
		args.eval_batch_size,
		shuffle=False,
		drop_last=False,
	)
	test_loader = make_loader(
		data_splits["test_images"],
		data_splits["test_labels"],
		args.eval_batch_size,
		shuffle=False,
		drop_last=False,
	)

	try:
		print("\n[1/3] Building classifier architecture...")
		graph_extended, node_labels, label_node_groups, arch_stats, arch_time = build_architecture(
			config, data_splits["num_classes"]
		)
		print(
			f"  Hidden={arch_stats['num_hidden']} | VH={arch_stats['vh']} | "
			f"VV={arch_stats['vv']} | HH={arch_stats['hh']} | arch_time={arch_time:.1f}s"
		)
	except Exception as exc:
		print(f"  Architecture construction FAILED: {exc}")
		return None

	try:
		print("\n[2/3] Training with validation-based early stopping...")
		model = graph_to_bm(graph_extended, node_labels)
		model.to(device)
		model, history, best_val_acc, best_val_loss = train_with_early_stopping(
			model,
			train_loader,
			val_loader,
			config,
			label_node_groups,
			data_splits["num_classes"],
		)
		_, final_val_acc, val_per_class = evaluate_classification_model(
			model,
			val_loader,
			label_node_groups,
			data_splits["num_classes"],
			config["nodes_per_label"],
		)
		test_loss, test_acc, test_per_class = evaluate_classification_model(
			model,
			test_loader,
			label_node_groups,
			data_splits["num_classes"],
			config["nodes_per_label"],
		)
		train_time = time.time() - trial_start - arch_time
		print(
			f"  Best epoch={history['best_epoch']} | best val_acc={best_val_acc:.4f} | "
			f"test_acc={test_acc:.4f}"
		)
	except Exception as exc:
		print(f"  Training or evaluation FAILED: {exc}")
		import traceback

		traceback.print_exc()
		return None

	total_time = time.time() - trial_start
	model_filename = (
		f"cls_hpsearch_{RUN_TIMESTAMP}_trial{trial_num:03d}_{config['architecture']}_"
		f"val{100.0 * best_val_acc:.2f}_test{100.0 * test_acc:.2f}.pt"
	)
	model_path = os.path.join(DATA_DIR, model_filename)

	print("\n[3/3] Saving checkpoint...")
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"training_history": history,
			"hyperparams": config,
			"graph_edges": list(graph_extended.edges()),
			"node_labels": node_labels,
			"label_node_groups": label_node_groups,
			"arch_stats": arch_stats,
			"digit_filter": data_splits["digit_filter"],
			"num_classes": data_splits["num_classes"],
			"metrics": {
				"best_val_acc": best_val_acc,
				"best_val_loss": best_val_loss,
				"final_val_acc": final_val_acc,
				"test_loss": test_loss,
				"test_acc": test_acc,
				"val_per_class": val_per_class,
				"test_per_class": test_per_class,
			},
		},
		model_path,
	)

	result = {
		"trial": trial_num,
		"timestamp": datetime.now().isoformat(),
		"model_path": model_path,
		"config": config,
		"arch_stats": arch_stats,
		"best_val_acc": best_val_acc,
		"best_val_loss": best_val_loss,
		"final_val_acc": final_val_acc,
		"test_loss": test_loss,
		"test_acc": test_acc,
		"best_epoch": history["best_epoch"],
		"stopped_early": history["stopped_early"],
		"total_time_seconds": total_time,
		"val_per_class": val_per_class,
		"test_per_class": test_per_class,
		"final_train_total_loss": history["train_total_loss"][-1] if history["train_total_loss"] else None,
	}

	print(f"\n  Trial {trial_num} complete in {total_time:.1f}s")
	print(
		f"  best_val_acc={best_val_acc:.4f} | best_val_loss={best_val_loss:.4f} | "
		f"test_acc={test_acc:.4f}"
	)

	del model
	if device.type == "cuda":
		torch.cuda.empty_cache()

	return result


def maybe_shorten_trial(config: dict, remaining_seconds: float):
	estimated_minutes = config["epochs"] * 2 + 8
	if remaining_seconds < estimated_minutes * 60 * 0.5:
		new_epochs = max(args.min_epochs, min(config["epochs"], int(max(remaining_seconds, 1) / 180)))
		config["epochs"] = max(args.min_epochs, new_epochs)
		print(f"  Reduced epochs to {config['epochs']} due to remaining time budget")

	max_label_time = max(1.0, remaining_seconds * 0.25)
	if config["label_assignment_time_limit"] > max_label_time:
		config["label_assignment_time_limit"] = max_label_time
		print(
			f"  Reduced label assignment time limit to "
			f"{config['label_assignment_time_limit']:.1f}s due to remaining time budget"
		)


def print_leaderboard(leaderboard: list):
	print(f"\n  {'=' * 62}")
	print(f"  TOP-{TOP_N} LEADERBOARD")
	print(f"  {'=' * 62}")
	for rank, entry in enumerate(leaderboard, start=1):
		cfg = entry["config"]
		print(
			f"  {rank}. Trial {entry['trial']:3d} | val_acc={100.0 * entry['best_val_acc']:.2f}% | "
			f"test_acc={100.0 * entry['test_acc']:.2f}% | {cfg['architecture']:8s} | "
			f"lr={cfg['lr']:.1e} | ep={cfg['epochs']} | k={cfg['k_steps']}"
		)


def main():
	data_splits = build_data_splits()
	precompute_architecture_assignments()

	if args.resume and os.path.exists(args.resume):
		leaderboard = load_leaderboard(args.resume)
		leaderboard_path = args.resume
		print(f"\nResumed leaderboard from {args.resume} ({len(leaderboard)} entries)")
		trial_num = max((entry.get("trial", 0) for entry in leaderboard), default=0)
	else:
		leaderboard = []
		leaderboard_path = LEADERBOARD_PATH
		trial_num = 0

	search_start = time.time()
	completed_trials = 0
	failed_trials = 0

	print(f"\n{'#' * 70}")
	print(f"  SEARCH STARTING — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"  Budget: {args.hours}h = {TIME_BUDGET_SECONDS:.0f}s")
	print(f"{'#' * 70}")

	while True:
		elapsed = time.time() - search_start
		remaining = TIME_BUDGET_SECONDS - elapsed
		attempted_trials = completed_trials + failed_trials

		if elapsed >= TIME_BUDGET_SECONDS:
			print(f"\nTime budget exhausted ({args.hours}h)")
			break
		if attempted_trials >= MAX_TRIALS:
			print(f"\nMax trials reached ({MAX_TRIALS})")
			break

		trial_num += 1
		print(f"\n{'-' * 70}")
		print(
			f"  Elapsed: {elapsed / 3600:.2f}h / {args.hours}h | "
			f"Remaining: {remaining / 3600:.2f}h | "
			f"Trials: {completed_trials} done, {failed_trials} failed"
		)
		if leaderboard:
			print(
				f"  Current best: val_acc={100.0 * leaderboard[0]['best_val_acc']:.2f}% "
				f"(trial {leaderboard[0]['trial']}, {leaderboard[0]['config']['architecture']})"
			)
		print(f"{'-' * 70}")

		config = sample_hyperparams()
		maybe_shorten_trial(config, remaining)

		result = run_trial(trial_num, config, data_splits)
		if result is None:
			failed_trials += 1
			continue

		completed_trials += 1
		leaderboard = update_leaderboard(leaderboard, result, TOP_N)
		save_leaderboard(leaderboard, leaderboard_path)
		print(f"\n  Leaderboard updated: {leaderboard_path}")
		print_leaderboard(leaderboard)

	total_time = time.time() - search_start
	print(f"\n{'#' * 70}")
	print(f"  SEARCH COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"  Total time: {total_time / 3600:.2f}h")
	print(f"  Trials: {completed_trials} completed, {failed_trials} failed")
	print(f"  Leaderboard: {leaderboard_path}")
	print(f"{'#' * 70}")

	if leaderboard:
		print_leaderboard(leaderboard)


if __name__ == "__main__":
	main()
