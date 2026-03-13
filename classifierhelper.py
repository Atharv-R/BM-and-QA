"""
ARCH-CLS: Discriminative Boltzmann Machine for MNIST Classification

Strategy: Extend the BM architecture by adding label nodes as additional visible units.
During training, both pixels AND labels are visible (clamped). During inference, we
marginalize over label nodes to perform classification via maximum likelihood.

Architecture:
    [144 pixel nodes] ←→ [H hidden nodes] ←→ [2 label nodes]
    
All are fully visible during training. For classification, we:
    1. Clamp pixel values
    2. Run Gibbs sampling over hidden + label nodes
    3. Take majority vote on label nodes

Rationale: Classification is easier than generation. By including labels in the
energy function, the model learns a discriminative boundary directly. This is
conceptually similar to conditional RBMs but uses the full BM framework.
"""

import os
import time
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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
    GRID_SHAPE,
    device,
    get_zephyr_positions
)


from optihelper import visualize_node_assignment_on_zephyr

ARCH_NAME = "classifier"
ARCH_LABEL = "ARCH-CLS: Discriminative BM"


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


def add_label_nodes_to_graph(G, visible_nodes, hidden_nodes, num_classes=10, nodes_per_label=1):
    """
    Extend the graph by adding label nodes connected to ALL hidden nodes.
    
    Args:
        G: Original Zephyr graph
        visible_nodes: List of pixel node IDs
        hidden_nodes: List of hidden node IDs
        num_classes: Number of classes (e.g., 10 for MNIST)
        nodes_per_label: How many nodes to allocate per class (1, 3, 5, etc.)
    
    Returns:
        G_extended: New graph with label nodes added
        label_node_groups: List of lists, where label_node_groups[c] contains 
                          node IDs for class c (length = nodes_per_label)
        node_labels_extended: Updated node_labels dict
    """
    G_extended = G.copy()
    
    num_pixels = len(visible_nodes)
    num_hidden = len(hidden_nodes)
    total_label_nodes = num_classes * nodes_per_label
    
    # Label nodes come after pixels and hidden nodes
    label_node_start = num_pixels + num_hidden
    all_label_nodes = list(range(label_node_start, label_node_start + total_label_nodes))
    
    # Group label nodes by class
    label_node_groups = []
    for c in range(num_classes):
        class_nodes = all_label_nodes[c * nodes_per_label : (c + 1) * nodes_per_label]
        label_node_groups.append(class_nodes)
    
    print(f"\nAdding {total_label_nodes} label nodes ({nodes_per_label} per class)...")
    print(f"  Pixel nodes: 0 to {num_pixels-1}")
    print(f"  Hidden nodes: {num_pixels} to {num_pixels+num_hidden-1}")
    print(f"  Label nodes: {label_node_start} to {label_node_start+total_label_nodes-1}")
    print(f"  Label grouping: {num_classes} classes × {nodes_per_label} nodes/class")
    
    # Add label nodes to graph
    for label_node in all_label_nodes:
        G_extended.add_node(label_node)
    
    # Connect each label node to ALL hidden nodes
    edges_added = 0
    for label_node in all_label_nodes:
        for hidden_node in hidden_nodes:
            G_extended.add_edge(label_node, hidden_node)
            edges_added += 1
    
    print(f"  Added {edges_added} label-hidden edges")
    
    # Update node labels: label nodes are ALSO visible during training
    node_labels_extended = {}
    for node in visible_nodes:
        node_labels_extended[node] = 'visible'
    for node in hidden_nodes:
        node_labels_extended[node] = 'hidden'
    for node in all_label_nodes:
        node_labels_extended[node] = 'visible'
    
    return G_extended, label_node_groups, node_labels_extended


def _get_label_coverage_targets(visible_nodes, hidden_nodes, coverage_mode='hidden_visible'):
    """Return the target node set used when scoring label-node coverage."""
    if coverage_mode == 'hidden_only':
        return set(hidden_nodes)
    if coverage_mode == 'hidden_visible':
        return set(hidden_nodes) | set(visible_nodes)
    raise ValueError(
        f"Unknown coverage_mode={coverage_mode!r}. "
        "Use 'hidden_only' or 'hidden_visible'."
    )


def _compute_candidate_coverage_sets(G, candidate_nodes, target_nodes):
    """Precompute which target nodes each candidate label node can directly influence."""
    coverage_sets = {}
    for node in candidate_nodes:
        coverage_sets[node] = set(G.neighbors(node)) & target_nodes
    return coverage_sets


def _score_label_assignment(label_node_groups, coverage_sets):
    """
    Score an assignment lexicographically.

    Primary objective: maximize the minimum per-class coverage.
    Secondary objectives: maximize total coverage across classes and reduce spread.
    """
    per_class_coverages = []
    per_class_nodes = []
    for group in label_node_groups:
        covered_nodes = set()
        for node in group:
            covered_nodes |= coverage_sets[node]
        per_class_nodes.append(covered_nodes)
        per_class_coverages.append(len(covered_nodes))

    min_coverage = min(per_class_coverages) if per_class_coverages else 0
    total_coverage = sum(per_class_coverages)
    spread = (max(per_class_coverages) - min_coverage) if per_class_coverages else 0
    score = (min_coverage, total_coverage, -spread)
    return score, per_class_coverages, per_class_nodes


def _greedy_initialize_label_groups(
    G,
    hidden_nodes,
    coverage_sets,
    num_classes,
    nodes_per_label,
):
    """Build an initial disjoint label assignment with round-robin greedy coverage."""
    available_nodes = set(hidden_nodes)
    label_node_groups = [[] for _ in range(num_classes)]
    covered_by_class = [set() for _ in range(num_classes)]

    for _ in range(nodes_per_label):
        class_order = sorted(range(num_classes), key=lambda idx: (len(covered_by_class[idx]), idx))
        for class_idx in class_order:
            best_node = None
            best_key = None
            for node in available_nodes:
                node_coverage = coverage_sets[node]
                marginal_gain = len(node_coverage - covered_by_class[class_idx])
                candidate_key = (marginal_gain, len(node_coverage), G.degree(node), -int(node))
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_node = node

            if best_node is None:
                raise RuntimeError("Ran out of candidate hidden nodes while initializing label groups.")

            label_node_groups[class_idx].append(best_node)
            covered_by_class[class_idx] |= coverage_sets[best_node]
            available_nodes.remove(best_node)

    return label_node_groups


def optimize_native_label_node_groups(
    G,
    visible_nodes,
    hidden_nodes,
    num_classes=10,
    nodes_per_label=1,
    coverage_mode='hidden_visible',
    time_limit_seconds=120.0,
    seed=None,
    verbose=True,
):
    """
    Select label nodes from the existing hidden-node pool without modifying the graph.

    The assignment is structural only: it does not use class-conditioned data.
    It maximizes the minimum per-class neighborhood coverage under the chosen
    coverage mode, with a greedy initialization followed by a time-limited local
    search that replaces weak label nodes using unassigned hidden candidates.
    """
    total_label_nodes = num_classes * nodes_per_label
    if total_label_nodes <= 0:
        raise ValueError("num_classes * nodes_per_label must be positive.")
    if total_label_nodes > len(hidden_nodes):
        raise ValueError(
            f"Need {total_label_nodes} native label nodes, but only {len(hidden_nodes)} hidden nodes are available."
        )

    target_nodes = _get_label_coverage_targets(visible_nodes, hidden_nodes, coverage_mode=coverage_mode)
    coverage_sets = _compute_candidate_coverage_sets(G, hidden_nodes, target_nodes)
    label_node_groups = _greedy_initialize_label_groups(
        G,
        hidden_nodes,
        coverage_sets,
        num_classes,
        nodes_per_label,
    )

    rng = np.random.default_rng(seed)
    current_groups = [group.copy() for group in label_node_groups]
    current_score, current_per_class_coverages, _ = _score_label_assignment(current_groups, coverage_sets)
    current_assigned = {node for group in current_groups for node in group}

    best_groups = [group.copy() for group in current_groups]
    best_score = current_score
    best_per_class_coverages = list(current_per_class_coverages)

    search_start = time.time()
    iterations = 0
    improvements = 0

    while time.time() - search_start < time_limit_seconds:
        iterations += 1
        worst_coverage = min(current_per_class_coverages)
        worst_classes = [
            class_idx for class_idx, cov in enumerate(current_per_class_coverages)
            if cov == worst_coverage
        ]
        focus_class = int(rng.choice(worst_classes))

        focus_nodes = current_groups[focus_class]
        focus_coverage = set()
        for node in focus_nodes:
            focus_coverage |= coverage_sets[node]

        available_nodes = [node for node in hidden_nodes if node not in current_assigned]
        if not available_nodes:
            break

        candidate_pool = sorted(
            available_nodes,
            key=lambda node: (
                len(coverage_sets[node] - focus_coverage),
                len(coverage_sets[node]),
                G.degree(node),
                -int(node),
            ),
            reverse=True,
        )[:64]

        best_move = None
        best_move_score = current_score
        best_move_per_class = current_per_class_coverages

        for node_idx, old_node in enumerate(focus_nodes):
            reduced_group = [node for idx, node in enumerate(focus_nodes) if idx != node_idx]
            reduced_coverage = set()
            for node in reduced_group:
                reduced_coverage |= coverage_sets[node]

            for new_node in candidate_pool:
                new_focus_coverage = reduced_coverage | coverage_sets[new_node]
                new_per_class_coverages = list(current_per_class_coverages)
                new_per_class_coverages[focus_class] = len(new_focus_coverage)
                min_cov = min(new_per_class_coverages)
                total_cov = sum(new_per_class_coverages)
                spread = max(new_per_class_coverages) - min_cov
                new_score = (min_cov, total_cov, -spread)

                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_per_class = new_per_class_coverages
                    best_move = (focus_class, node_idx, old_node, new_node)

        if best_move is None:
            continue

        class_idx, node_idx, old_node, new_node = best_move
        current_groups[class_idx][node_idx] = new_node
        current_assigned.remove(old_node)
        current_assigned.add(new_node)
        current_score = best_move_score
        current_per_class_coverages = best_move_per_class
        improvements += 1

        if current_score > best_score:
            best_score = current_score
            best_groups = [group.copy() for group in current_groups]
            best_per_class_coverages = list(current_per_class_coverages)

    label_nodes_flat = [node for group in best_groups for node in group]
    score, per_class_coverages, per_class_nodes = _score_label_assignment(best_groups, coverage_sets)
    assignment_stats = {
        'coverage_mode': coverage_mode,
        'objective_score': {
            'min_class_coverage': score[0],
            'total_class_coverage': score[1],
            'coverage_spread': -score[2],
        },
        'per_class_coverage': per_class_coverages,
        'selected_label_nodes_original': best_groups,
        'selected_label_nodes_flat_original': label_nodes_flat,
        'num_label_nodes': len(label_nodes_flat),
        'iterations': iterations,
        'improvements': improvements,
        'time_seconds': time.time() - search_start,
        'covered_nodes_per_class_original': [sorted(nodes) for nodes in per_class_nodes],
    }

    if verbose:
        print(
            f"\nNative label-node assignment complete: min coverage={score[0]}, "
            f"total coverage={score[1]}, spread={-score[2]}, "
            f"iterations={iterations}, improvements={improvements}, "
            f"time={assignment_stats['time_seconds']:.1f}s"
        )

    return best_groups, assignment_stats


def relabel_graph_with_label_nodes(G, pixel_visible_nodes, hidden_nodes, label_node_groups):
    """
    Relabel an existing Zephyr-native graph so the visible ordering becomes:
    pixel visibles -> native label visibles -> remaining hidden nodes.

    No nodes or edges are added or removed; only node ids are remapped.
    """
    label_nodes_flat = [node for group in label_node_groups for node in group]
    label_node_set = set(label_nodes_flat)

    if len(label_node_set) != len(label_nodes_flat):
        raise ValueError("Label node groups must be disjoint.")
    if not label_node_set.issubset(set(hidden_nodes)):
        raise ValueError("All native label nodes must come from the hidden-node pool.")

    mapping = {}
    for new_idx, old_node in enumerate(pixel_visible_nodes):
        mapping[old_node] = new_idx

    next_visible_idx = len(pixel_visible_nodes)
    relabeled_label_groups = []
    for group in label_node_groups:
        relabeled_group = []
        for old_node in group:
            mapping[old_node] = next_visible_idx
            relabeled_group.append(next_visible_idx)
            next_visible_idx += 1
        relabeled_label_groups.append(relabeled_group)

    remaining_hidden_nodes = [node for node in hidden_nodes if node not in label_node_set]
    hidden_offset = next_visible_idx
    for offset, old_node in enumerate(remaining_hidden_nodes):
        mapping[old_node] = hidden_offset + offset

    G_relabeled = nx.relabel_nodes(G, mapping, copy=True)

    node_labels = {}
    for node in range(len(pixel_visible_nodes)):
        node_labels[node] = 'visible'
    for group in relabeled_label_groups:
        for node in group:
            node_labels[node] = 'visible'
    for node in range(hidden_offset, hidden_offset + len(remaining_hidden_nodes)):
        node_labels[node] = 'hidden'

    relabeled_remaining_hidden = list(range(hidden_offset, hidden_offset + len(remaining_hidden_nodes)))
    return G_relabeled, relabeled_label_groups, relabeled_remaining_hidden, node_labels, mapping


def build_native_labelled_graph(
    G,
    pixel_visible_nodes,
    hidden_nodes,
    num_classes=10,
    nodes_per_label=1,
    coverage_mode='hidden_visible',
    time_limit_seconds=120.0,
    seed=None,
    verbose=True,
):
    """
    Select native Zephyr label nodes from the hidden pool and relabel the graph
    into a contiguous visible/hidden layout suitable for the BM implementation.
    """
    label_node_groups_original, assignment_stats = optimize_native_label_node_groups(
        G,
        pixel_visible_nodes,
        hidden_nodes,
        num_classes=num_classes,
        nodes_per_label=nodes_per_label,
        coverage_mode=coverage_mode,
        time_limit_seconds=time_limit_seconds,
        seed=seed,
        verbose=verbose,
    )

    G_relabeled, label_node_groups, remaining_hidden_nodes, node_labels, mapping = (
        relabel_graph_with_label_nodes(
            G,
            pixel_visible_nodes,
            hidden_nodes,
            label_node_groups_original,
        )
    )

    assignment_stats.update({
        'selected_label_nodes_relabeled': label_node_groups,
        'remaining_hidden_nodes_relabeled': remaining_hidden_nodes,
        'num_remaining_hidden_nodes': len(remaining_hidden_nodes),
    })

    return G_relabeled, label_node_groups, remaining_hidden_nodes, node_labels, assignment_stats, mapping


def prepare_classification_batch(pixels, labels, label_node_groups, num_classes=10, nodes_per_label=1):
    """
    Convert (pixels, labels) batch into extended visible vector for training.
    Now handles multiple nodes per label.
    
    Args:
        pixels: (batch_size, num_pixels) tensor
        labels: (batch_size,) tensor with class indices
        label_node_groups: List of lists, where label_node_groups[c] contains node IDs for class c
        num_classes: Number of classes
        nodes_per_label: How many nodes per class
    
    Returns:
        extended_visible: (batch_size, num_pixels + num_classes*nodes_per_label) tensor
    """
    batch_size = pixels.shape[0]
    total_label_nodes = num_classes * nodes_per_label
    
    # Verify labels are in valid range
    assert labels.min() >= 0 and labels.max() < num_classes, \
        f"Labels out of range! Got [{labels.min()}, {labels.max()}], expected [0, {num_classes-1}]"
    
    # Create label representation: all nodes for the true class are set to 1, others to 0
    label_representation = torch.zeros(batch_size, total_label_nodes, device=pixels.device)
    
    for i in range(batch_size):
        true_class = labels[i].item()
        # Set all nodes for this class to 1
        start_idx = true_class * nodes_per_label
        end_idx = start_idx + nodes_per_label
        label_representation[i, start_idx:end_idx] = 1.0
    
    # Concatenate: [pixels | label_nodes]
    extended_visible = torch.cat([pixels, label_representation], dim=1)
    
    return extended_visible


def compute_class_scores_free_energy(model, pixels, label_node_groups, num_classes=10, nodes_per_label=1):
    """
    Score each class by negative free energy for the corresponding clamped label state.

    This matches the intended conditional-likelihood style classifier and remains
    compatible with QA-style inference, where labels can be clamped and scored via
    low-energy states of the BM.
    """
    class_scores = []
    for class_idx in range(num_classes):
        candidate_labels = torch.full(
            (pixels.shape[0],), class_idx, dtype=torch.long, device=pixels.device
        )
        candidate_visible = prepare_classification_batch(
            pixels, candidate_labels, label_node_groups, num_classes, nodes_per_label
        )
        class_scores.append(-model.free_energy(candidate_visible))

    return torch.stack(class_scores, dim=1)

def train_classifier_bm(model, data_loader, optimizer, num_epochs, k_steps, 
                        label_node_groups, batch_size, step_size, num_classes=10, nodes_per_label=1,
                        classification_loss_weight=1.0):
    """
    Train discriminative BM where labels are part of the visible layer.
    Now supports multiple nodes per label.
    """
    from bolmaqua import compute_pseudolikelihood, evaluate_reconstruction
    
    loss_history = []
    classification_loss_history = []
    total_loss_history = []
    pll_values = []
    train_recon_mse_history = []  
    train_recon_acc_history = []
    train_recon_bce_history = []
    
    print(f"\nTraining Classifier BM (epochs={num_epochs}, k_steps={k_steps}, "
          f"num_classes={num_classes}, nodes_per_label={nodes_per_label})...")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (pixels, labels) in enumerate(data_loader):
            pixels = pixels.to(device)
            labels = labels.to(device)
            
            # Extend visible to include labels (with multiple nodes per label)
            extended_visible = prepare_classification_batch(
                pixels, labels, label_node_groups, num_classes, nodes_per_label
            )
            
            optimizer.zero_grad(set_to_none=True)
            cd_loss, _ = model(extended_visible, k_steps=k_steps)
            class_scores = compute_class_scores_free_energy(
                model, pixels, label_node_groups, num_classes, nodes_per_label
            )
            cls_loss = F.cross_entropy(class_scores, labels)
            total_loss = cd_loss + classification_loss_weight * cls_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += cd_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

            del extended_visible, cd_loss, cls_loss, class_scores, total_loss, pixels, labels
        
        avg_loss = epoch_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        loss_history.append(avg_loss)
        classification_loss_history.append(avg_cls_loss)
        total_loss_history.append(avg_total_loss)
        
        # Compute PLL
        with torch.no_grad():
            sample_batch = next(iter(data_loader))
            sample_pixels, sample_labels = sample_batch[0].to(device), sample_batch[1].to(device)
            sample_extended = prepare_classification_batch(
                sample_pixels, sample_labels, label_node_groups, num_classes, nodes_per_label
            )
            pll = compute_pseudolikelihood(model, sample_extended, num_samples=50)
            pll_values.append(pll)

            train_metrics = evaluate_reconstruction(model, sample_extended, num_samples=50)
            train_recon_mse_history.append(train_metrics['mse'])
            train_recon_acc_history.append(train_metrics['accuracy'])
            train_recon_bce_history.append(train_metrics['bce'])
        
        print(f"Epoch {epoch+1}/{num_epochs} | CD: {avg_loss:.4f} | Cls: {avg_cls_loss:.4f} | "
              f"Total: {avg_total_loss:.4f} | PLL: {pll:.4f} | "
              f"Recon MSE: {train_recon_mse_history[-1]:.4f} | Recon Acc: {train_recon_acc_history[-1]:.4f} | "
              f"Recon BCE: {train_recon_bce_history[-1]:.4f}")
    
    return {
        'pcd_loss': loss_history,
        'classification_loss': classification_loss_history,
        'total_loss': total_loss_history,
        'pll': pll_values,
    }


def classify_images_fast(model, test_pixels, label_node_groups, num_gibbs_steps=50, 
                        num_classes=10, nodes_per_label=1, aggregation='average',
                        inference_method='free_energy'):
    """
    Classification using either free-energy scoring or the legacy mean-field path.
    
    Args:
        aggregation: 'average' (soft voting) or 'majority' (hard voting)
        inference_method: 'free_energy' or 'mean_field'
    """
    was_training = model.training
    model.eval()
    batch_size = test_pixels.shape[0]
    num_pixels = test_pixels.shape[1]
    total_label_nodes = num_classes * nodes_per_label
    
    with torch.inference_mode():
        if inference_method == 'free_energy':
            class_scores = compute_class_scores_free_energy(
                model, test_pixels, label_node_groups, num_classes, nodes_per_label
            )
            probabilities = torch.softmax(class_scores, dim=1)
            pred_classes = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
        elif inference_method == 'mean_field':
            # Initialize: clamp pixels, randomize labels and hidden
            extended_v = torch.cat([
                test_pixels,
                torch.rand(batch_size, total_label_nodes, device=device)
            ], dim=1)
            
            h = torch.rand(batch_size, model.num_hidden, device=device)
            
            # Run mean-field updates with pixels CLAMPED
            for step in range(num_gibbs_steps):
                _, h = model.mean_field_update(extended_v, h, update_v=False, update_h=True)
                v_recon = extended_v.clone()
                v_recon, _ = model.mean_field_update(v_recon, h, update_v=True, update_h=False)
                extended_v[:, num_pixels:] = v_recon[:, num_pixels:]
            
            # Extract label node activations
            label_activations = extended_v[:, num_pixels:]
            class_scores = torch.zeros(batch_size, num_classes, device=device)
            
            if aggregation == 'average':
                for c in range(num_classes):
                    start_idx = c * nodes_per_label
                    end_idx = start_idx + nodes_per_label
                    class_scores[:, c] = label_activations[:, start_idx:end_idx].mean(dim=1)
            elif aggregation == 'majority':
                binary_activations = (label_activations > 0.5).float()
                for c in range(num_classes):
                    start_idx = c * nodes_per_label
                    end_idx = start_idx + nodes_per_label
                    class_scores[:, c] = binary_activations[:, start_idx:end_idx].sum(dim=1)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            pred_classes = torch.argmax(class_scores, dim=1)
            confidences = torch.max(class_scores, dim=1)[0]
            if aggregation == 'majority':
                confidences = confidences / nodes_per_label
        else:
            raise ValueError(f"Unknown inference method: {inference_method}")
        
    if was_training:
        model.train()
    
    return pred_classes, confidences


def evaluate_classifier(model, test_loader, label_node_groups, num_gibbs_steps=100, 
                       num_classes=10, nodes_per_label=1, aggregation='average',
                       inference_method='free_energy'):
    """
    Evaluate classification accuracy with ensemble voting.
    """
    was_training = model.training
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\nEvaluating classifier (method={inference_method}, steps={num_gibbs_steps}, "
          f"nodes_per_label={nodes_per_label}, aggregation={aggregation})...")
    
    with torch.inference_mode():
        for batch_idx, (pixels, labels) in enumerate(test_loader):
            pixels = pixels.to(device)
            labels = labels.to(device)
            
            preds, confs = classify_images_fast(
                model, pixels, label_node_groups, 
                num_gibbs_steps=num_gibbs_steps,
                num_classes=num_classes,
                nodes_per_label=nodes_per_label,
                aggregation=aggregation,
                inference_method=inference_method,
            )
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            if (batch_idx + 1) % max(1, len(test_loader) // 10) == 0:
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
    
    print(f"\n  Overall Accuracy: {100*accuracy:.2f}%")
    for c in range(num_classes):
        if c in per_class_acc:
            print(f"  Class {c} Accuracy: {100*per_class_acc[c]:.2f}%")

    if was_training:
        model.train()
    
    return accuracy, per_class_acc

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