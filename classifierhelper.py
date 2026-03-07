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

def train_classifier_bm(model, data_loader, optimizer, num_epochs, k_steps, 
                        label_node_groups, batch_size, step_size, num_classes=10, nodes_per_label=1):
    """
    Train discriminative BM where labels are part of the visible layer.
    Now supports multiple nodes per label.
    """
    from bolmaqua import compute_pseudolikelihood, evaluate_reconstruction
    
    loss_history = []
    pll_values = []
    train_recon_mse_history = []  
    train_recon_acc_history = []
    train_recon_bce_history = []
    
    print(f"\nTraining Classifier BM (epochs={num_epochs}, k_steps={k_steps}, "
          f"num_classes={num_classes}, nodes_per_label={nodes_per_label})...")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (pixels, labels) in enumerate(data_loader):
            pixels = pixels.to(device)
            labels = labels.to(device)
            
            # Extend visible to include labels (with multiple nodes per label)
            extended_visible = prepare_classification_batch(
                pixels, labels, label_node_groups, num_classes, nodes_per_label
            )
            
            optimizer.zero_grad()
            loss, _ = model(extended_visible, k_steps=k_steps)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
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
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | PLL: {pll:.4f} | "
              f"Recon MSE: {train_recon_mse_history[-1]:.4f} | Recon Acc: {train_recon_acc_history[-1]:.4f} | "
              f"Recon BCE: {train_recon_bce_history[-1]:.4f}")
    
    return {'pcd_loss': loss_history, 'pll': pll_values}


def classify_images_fast(model, test_pixels, label_node_groups, num_gibbs_steps=50, 
                        num_classes=10, nodes_per_label=1, aggregation='average'):
    """
    FAST classification using mean-field inference with ensemble voting.
    
    Args:
        aggregation: 'average' (soft voting) or 'majority' (hard voting)
    """
    model.eval()
    batch_size = test_pixels.shape[0]
    num_pixels = test_pixels.shape[1]
    total_label_nodes = num_classes * nodes_per_label
    
    with torch.no_grad():
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
        label_activations = extended_v[:, num_pixels:]  # (batch_size, total_label_nodes)
        
        # Aggregate across nodes for each class
        class_scores = torch.zeros(batch_size, num_classes, device=device)
        
        if aggregation == 'average':
            # Soft voting: average activations of nodes for each class
            for c in range(num_classes):
                start_idx = c * nodes_per_label
                end_idx = start_idx + nodes_per_label
                class_scores[:, c] = label_activations[:, start_idx:end_idx].mean(dim=1)
        
        elif aggregation == 'majority':
            # Hard voting: binarize each node, then majority vote
            binary_activations = (label_activations > 0.5).float()
            for c in range(num_classes):
                start_idx = c * nodes_per_label
                end_idx = start_idx + nodes_per_label
                # Count how many nodes vote for this class
                class_scores[:, c] = binary_activations[:, start_idx:end_idx].sum(dim=1)
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        pred_classes = torch.argmax(class_scores, dim=1)
        confidences = torch.max(class_scores, dim=1)[0]
        
        # Normalize confidences to [0, 1]
        if aggregation == 'majority':
            confidences = confidences / nodes_per_label
        # For 'average', confidences are already in [0, 1]
    
    return pred_classes, confidences


def evaluate_classifier(model, test_loader, label_node_groups, num_gibbs_steps=100, 
                       num_classes=10, nodes_per_label=1, aggregation='average'):
    """
    Evaluate classification accuracy with ensemble voting.
    """
    all_preds = []
    all_labels = []
    
    print(f"\nEvaluating classifier (mean-field steps={num_gibbs_steps}, "
          f"nodes_per_label={nodes_per_label}, aggregation={aggregation})...")
    
    for batch_idx, (pixels, labels) in enumerate(test_loader):
        pixels = pixels.to(device)
        labels = labels.to(device)
        
        preds, confs = classify_images_fast(
            model, pixels, label_node_groups, 
            num_gibbs_steps=num_gibbs_steps,
            num_classes=num_classes,
            nodes_per_label=nodes_per_label,
            aggregation=aggregation
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