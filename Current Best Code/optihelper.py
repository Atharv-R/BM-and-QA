# pip install pyscipopt   
import dwave_networkx as dnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import resize
import pandas as pd
import os
from torchvision import datasets, transforms
import dwave.system
import dimod
from dwave.samplers import SimulatedAnnealingSampler

from pyscipopt import Model, quicksum
import numpy as np
import dimod
import itertools
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, SolverFactory, NonNegativeIntegers, Binary, summation, value, RangeSet


# Analysis HELPERS
def analyze_zephyr_layout(G, K, save_to_csv=True):
    """
    Detailed analysis of Zephyr graph structure.
    Returns DataFrame with node positions, degrees, and identifies edge nodes.
    """
    import pandas as pd
    
    # Get Zephyr layout positions
    try:
        pos = dnx.zephyr_layout(G)
    except:
        print("⚠️ Using spring layout fallback")
        pos = nx.spring_layout(G, seed=42)
    
    # Collect node data
    node_data = []
    for node in G.nodes():
        x, y = pos[node]
        deg = G.degree(node)
        
        # Calculate distance from center
        center_x = sum(p[0] for p in pos.values()) / len(pos)
        center_y = sum(p[1] for p in pos.values()) / len(pos)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Identify if it's an edge node 
        is_edge = (deg < 8) or (dist_from_center > np.percentile([
            np.sqrt((pos[n][0] - center_x)**2 + (pos[n][1] - center_y)**2) 
            for n in G.nodes()
        ], 80))
        
        node_data.append({
            'node': node,
            'x': x,
            'y': y,
            'degree': deg,
            'dist_from_center': dist_from_center,
            'is_edge': is_edge
        })
    
    df = pd.DataFrame(node_data)
    
    # Sort by different criteria
    print("\n Top 20 nodes by degree ")
    print(df.nlargest(20, 'degree')[['node', 'degree', 'x', 'y', 'dist_from_center']])
    
    print("\n Bottom 20 nodes by degree (edge candidates) ")
    print(df.nsmallest(20, 'degree')[['node', 'degree', 'x', 'y', 'dist_from_center']])
    
    print("\n 20 nodes farthest from center ")
    print(df.nlargest(20, 'dist_from_center')[['node', 'degree', 'x', 'y', 'dist_from_center']])
    
    print("\nDegree distribution ")
    print(df['degree'].value_counts().sort_index())
    
    print(f"\nEdge node statistics")
    edge_nodes = df[df['is_edge']]
    print(f"  Edge nodes identified: {len(edge_nodes)}")
    print(f"  Avg degree (edge): {edge_nodes['degree'].mean():.2f}")
    print(f"  Avg degree (center): {df[~df['is_edge']]['degree'].mean():.2f}")
    
    if save_to_csv:
        filename = f"zephyr_k{K}_node_analysis.csv"
        df.to_csv(filename, index=False)
        print(f"\n Saved full analysis to {filename}")
    
    return df



def visualize_node_assignment_on_zephyr(G, visible_nodes, hidden_nodes, title="Node Assignment"):
    """
    Visualize which nodes are visible/hidden overlaid on Zephyr structure.
    Color by degree and assignment.
    """
    try:
        pos = dnx.zephyr_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42)
    
    # Prepare colors and sizes
    colors = []
    sizes = []
    labels_to_show = {}
    
    for node in G.nodes():
        deg = G.degree(node)
        
        if node in visible_nodes:
            colors.append('red')
            sizes.append(100 + deg * 10)  # Larger = higher degree
        elif node in hidden_nodes:
            colors.append('blue')
            sizes.append(100 + deg * 10)
        else:
            colors.append('gray')
            sizes.append(50)
        
        # Label only low-degree nodes for debugging
        if deg < 6:
            labels_to_show[node] = f"{node}\n(d={deg})"
    
    plt.figure(figsize=(16, 12))
    
    # Draw all edges lightly
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='lightgray', width=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.7)
    
    # Draw labels for low-degree nodes
    nx.draw_networkx_labels(G, pos, labels=labels_to_show, font_size=6)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Visible'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='Hidden'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Removed')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"{title}\n(Node size ∝ degree, labels show degree < 6)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Pixel-Visible node assignment (heuristic)

def compute_pixel_importance(grid_shape=(12, 12)):
    """
    Assign importance scores to pixels based on distance from center.
    Returns: importance_map (grid_shape), flat_importance (flattened)
    """
    rows, cols = grid_shape
    center_r, center_c = rows / 2, cols / 2
    
    importance_map = np.zeros(grid_shape)
    for r in range(rows):
        for c in range(cols):
            # Gaussian-like importance: higher at center
            dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
            importance_map[r, c] = np.exp(-dist**2 / (2 * (rows/4)**2))
    
    # Normalize to [0, 1]
    importance_map = importance_map / importance_map.max()
    flat_importance = importance_map.flatten()
    
    return importance_map, flat_importance

def compute_pixel_adjacency(grid_shape=(12, 12)):
    """
    Build adjacency list for pixels (8-connectivity or 4-connectivity).
    Returns: dict mapping pixel_id -> set of neighbor pixel_ids
    """
    rows, cols = grid_shape
    pixel_adj = {}
    
    for r in range(rows):
        for c in range(cols):
            pid = r * cols + c
            neighbors = set()
            
            # 4-connectivity (up, down, left, right)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.add(nr * cols + nc)
            
            # Optional: add diagonal neighbors for 8-connectivity
            # for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            #     nr, nc = r + dr, c + dc
            #     if 0 <= nr < rows and 0 <= nc < cols:
            #         neighbors.add(nr * cols + nc)
            
            pixel_adj[pid] = neighbors
    
    return pixel_adj

def assign_pixels_to_nodes_heuristic(G, num_pixels=144, grid_shape=(12, 12)):
    """
    Heuristically assign graph nodes to pixels, prioritizing:
    1. High-degree nodes for center pixels (high importance)
    2. Spatially close pixels should map to graph-neighboring nodes
    
    Returns: pixel_to_node (dict: pixel_id -> node_id)
    """
    importance_map, flat_importance = compute_pixel_importance(grid_shape)
    pixel_adj = compute_pixel_adjacency(grid_shape)
    
    # Sort pixels by importance (center first)
    pixel_order = np.argsort(-flat_importance)  # Descending importance
    
    # Sort nodes by degree (high degree first)
    nodes = sorted(G.nodes())
    node_degrees = {n: G.degree(n) for n in nodes}
    node_order = sorted(nodes, key=lambda n: -node_degrees[n])
    
    pixel_to_node = {}
    node_to_pixel = {}
    used_nodes = set()
    
    # Greedy assignment with spatial awareness
    for pixel_id in pixel_order:
        if len(pixel_to_node) >= num_pixels:
            break
        
        # Get already-assigned neighbor pixels
        assigned_neighbors = [pixel_to_node[nb] for nb in pixel_adj[pixel_id] 
                             if nb in pixel_to_node]
        
        # Find candidate nodes that are:
        # 1. Not yet used
        # 2. Preferably connected to nodes of neighbor pixels
        candidate_nodes = [n for n in node_order if n not in used_nodes]
        
        if assigned_neighbors:
            # Score nodes by how many graph-neighbors they share with assigned pixel neighbors
            def score_node(node):
                graph_neighbors = set(G.neighbors(node))
                overlap = len(graph_neighbors & set(assigned_neighbors))
                return overlap + node_degrees[node] * 0.1  # Slight degree bonus
            
            candidate_nodes.sort(key=score_node, reverse=True)
        
        # Assign best candidate
        chosen_node = candidate_nodes[0]
        pixel_to_node[pixel_id] = chosen_node
        node_to_pixel[chosen_node] = pixel_id
        used_nodes.add(chosen_node)
    
    print(f"Assigned {len(pixel_to_node)} pixels to nodes")
    print(f"  Center pixel (0,0) -> node {pixel_to_node.get(0, 'N/A')} with degree {node_degrees.get(pixel_to_node.get(0, -1), 0)}")
    
    return pixel_to_node, node_to_pixel


def preprocess_graph(G, V_target, H_target):
    """Remove low-value nodes before optimization to reduce problem size."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    
    # Calculate how many nodes to remove
    target_total = V_target + H_target
    nodes_to_remove = n - target_total
    
    # Remove lowest-degree nodes iteratively
    G_work = G.copy()
    removed = 0
    
    while removed < nodes_to_remove:
        degrees = [(node, G_work.degree(node)) for node in G_work.nodes()]
        degrees.sort(key=lambda x: x[1])
        
        # Remove lowest degree node
        node_to_remove = degrees[0][0]
        G_work.remove_node(node_to_remove)
        removed += 1
        
        if removed % 50 == 0:
            print(f"  Removed {removed}/{nodes_to_remove} nodes...")
    
    print(f"Preprocessed: {n} → {G_work.number_of_nodes()} nodes")
    return G_work


def remove_edge_nodes(G, min_degree=6):
    """
    Remove poorly-connected edge nodes from Zephyr graph.
    Zephyr K=3 has max degree ~8, so nodes with degree < 6 are edges.
    """
    G_clean = G.copy()
    nodes_to_remove = []
    
    for node in G.nodes():
        if G.degree(node) < min_degree:
            nodes_to_remove.append(node)
    
    print(f"Removing {len(nodes_to_remove)} edge nodes (degree < {min_degree})")
    G_clean.remove_nodes_from(nodes_to_remove)
    
    # Relabel nodes to be consecutive
    G_clean = nx.convert_node_labels_to_integers(G_clean, ordering='sorted')
    
    print(f"  Original nodes: {G.number_of_nodes()}")
    print(f"  Cleaned nodes: {G_clean.number_of_nodes()}")
    print(f"  Degree distribution after cleaning: min={min(dict(G_clean.degree()).values())}, max={max(dict(G_clean.degree()).values())}")
    
    return G_clean


# Analyze hidden-hidden connectivity
def analyze_fusion_potential(G, nodes, V_target):
    """Checks how many fusions are theoretically possible."""
    # Assume first V_target nodes will be visible (worst case for hidden connectivity)
    hidden_nodes = nodes[V_target:]
    
    # Build subgraph of just hidden nodes
    H_subgraph = G.subgraph(hidden_nodes)
    
    # Count connected components
    components = list(nx.connected_components(H_subgraph))
    num_components = len(components)
    
    # Maximum fusions = |hidden_nodes| - |components|
    # (you can't fuse nodes across disconnected components)
    max_fusions_possible = len(hidden_nodes) - num_components
    
    print(f"\n Fusion Potential Analysis ")
    print(f"  Hidden nodes: {len(hidden_nodes)}")
    print(f"  Connected components in hidden subgraph: {num_components}")
    print(f"  Maximum fusions possible: {max_fusions_possible}")
    print(f"  Required fusions: {len(hidden_nodes) - H_target}")
    print(f"  Gap: {max_fusions_possible - (len(hidden_nodes) - H_target)}")
    
    # Component sizes
    comp_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"  Component sizes: {comp_sizes[:10]}...")
    
    return max_fusions_possible


def apply_fusions_to_graph_clean(G_original, x_vals, z_vals):
    """
    Wrapper that ensures clean integer node IDs throughout.
    """
    # First, ensure input graph has integer nodes
    if not all(isinstance(n, int) for n in G_original.nodes()):
        print("⚠️ Input graph has non-integer nodes, relabeling...")
        mapping = {old: i for i, old in enumerate(sorted(G_original.nodes(), key=str))}
        G_original = nx.relabel_nodes(G_original, mapping)
        
        # Update x_vals and z_vals with new node IDs
        x_vals_new = {mapping.get(k, k): v for k, v in x_vals.items() if k in mapping}
        z_vals_new = {}
        for (k, l), v in z_vals.items():
            if k in mapping and l in mapping:
                k_new, l_new = mapping[k], mapping[l]
                z_vals_new[(min(k_new, l_new), max(k_new, l_new))] = v
        
        x_vals = x_vals_new
        z_vals = z_vals_new
    
    # Now apply fusions (using previous function logic)
    G = G_original.copy()
    nodes = sorted(G.nodes())  # Now safe to sort - all integers
    
    visible_nodes_original = set(i for i in nodes if x_vals.get(i, 0) > 0.5)
    hidden_nodes_original = set(i for i in nodes if x_vals.get(i, 0) <= 0.5)
    
    print(f"\n=== Pre-Fusion Status ===")
    print(f"  Visible nodes: {len(visible_nodes_original)}")
    print(f"  Hidden nodes: {len(hidden_nodes_original)}")
    
    # Union-Find for hidden nodes only
    parent = {i: i for i in hidden_nodes_original}
    
    def find(i):
        if i not in parent:
            return i
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    
    def union(i, j):
        if i not in hidden_nodes_original or j not in hidden_nodes_original:
            return False
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi
            return True
        return False
    
    fusions_applied = 0
    for (k, l), val in z_vals.items():
        if val > 0.5:
            if union(k, l):
                fusions_applied += 1
    
    print(f"  Fusions applied: {fusions_applied}")
    
    # Create fusion groups
    fusion_groups = {}
    for h_node in hidden_nodes_original:
        root = find(h_node)
        if root not in fusion_groups:
            fusion_groups[root] = []
        fusion_groups[root].append(h_node)
    
    # Contract graph
    G_contracted = G.copy()
    old_to_new = {}
    
    # Visible nodes unchanged
    for v in visible_nodes_original:
        old_to_new[v] = v
    
    # Fuse hidden nodes
    for root, group in fusion_groups.items():
        if len(group) > 1:
            representative = min(group)
            for node in group:
                old_to_new[node] = representative
            
            all_neighbors = set()
            for node in group:
                if node in G_contracted:
                    all_neighbors.update(G_contracted.neighbors(node))
            all_neighbors -= set(group)
            
            for node in group:
                if node != representative and node in G_contracted:
                    G_contracted.remove_node(node)
            
            for neighbor in all_neighbors:
                neighbor_mapped = old_to_new.get(neighbor, neighbor)
                if neighbor_mapped != representative:
                    G_contracted.add_edge(representative, neighbor_mapped)
        else:
            old_to_new[group[0]] = group[0]
    
    visible_nodes_final = sorted(visible_nodes_original)
    hidden_nodes_final = sorted(set(old_to_new[h] for h in hidden_nodes_original))
    
    # **KEY FIX: Relabel to consecutive integers**
    print("\n=== Relabeling to consecutive integers ===")
    all_nodes_ordered = visible_nodes_final + hidden_nodes_final
    relabel_map = {old_id: new_id for new_id, old_id in enumerate(all_nodes_ordered)}
    
    G_contracted = nx.relabel_nodes(G_contracted, relabel_map)
    visible_nodes_final = [relabel_map[v] for v in visible_nodes_final]
    hidden_nodes_final = [relabel_map[h] for h in hidden_nodes_final]
    
    print(f"  Relabeled nodes: 0 to {len(G_contracted.nodes())-1}")
    print(f"  Visible IDs: 0 to {len(visible_nodes_final)-1}")
    print(f"  Hidden IDs: {len(visible_nodes_final)} to {len(all_nodes_ordered)-1}")
    
    return G_contracted, visible_nodes_final, hidden_nodes_final, old_to_new


def postprocess_fuse_hidden(G_contracted, visible_nodes, hidden_nodes, H_target):
    """
    Stage 2: Greedily fuse ONLY hidden nodes to reach target.
    VISIBLE NODES ARE NEVER TOUCHED.
    """
    print(f"\n=== POST-PROCESSING: Fuse {len(hidden_nodes)} → {H_target} hidden nodes ===")
    
    if len(hidden_nodes) <= H_target:
        print(f"  Already at or below target!")
        return G_contracted, visible_nodes, hidden_nodes
    
    G_work = G_contracted.copy()
    visible_set = set(visible_nodes)  # Track which nodes are visible
    hidden_remaining = set(hidden_nodes)
    
    fusions_needed = len(hidden_nodes) - H_target
    fusions_done = 0
    
    while fusions_done < fusions_needed and len(hidden_remaining) > H_target:
        # Find lowest-degree HIDDEN node
        hidden_degrees = [(n, G_work.degree(n)) for n in hidden_remaining]
        if not hidden_degrees:
            break
        hidden_degrees.sort(key=lambda x: x[1])
        
        node1 = hidden_degrees[0][0]
        
        # Find its lowest-degree HIDDEN neighbor (don't fuse with visible!)
        hidden_neighbors = [nb for nb in G_work.neighbors(node1) 
                           if nb in hidden_remaining and nb != node1]
        
        if not hidden_neighbors:
            # Isolated hidden node - just remove it
            G_work.remove_node(node1)
            hidden_remaining.discard(node1)
            fusions_done += 1
            continue
        
        node2 = min(hidden_neighbors, key=lambda n: G_work.degree(n))
        
        # Contract node2 INTO node1 (node1 survives, node2 disappears)
        G_work = nx.contracted_nodes(G_work, node1, node2, self_loops=False)
        hidden_remaining.discard(node2)
        fusions_done += 1
        
        if fusions_done % 10 == 0:
            print(f"  Progress: {fusions_done}/{fusions_needed} fusions")
    
    # Final counts
    final_hidden = list(hidden_remaining)
    final_visible = [v for v in visible_nodes if v in G_work.nodes()]
    
    print(f"✓ Post-processing complete:")
    print(f"  Visible: {len(final_visible)} (target: {len(visible_nodes)})")
    print(f"  Hidden: {len(final_hidden)} (target: {H_target})")
    print(f"  Total nodes: {G_work.number_of_nodes()}")
    
    # Verify visible count
    if len(final_visible) != len(visible_nodes):
        print(f"  WARNING: Lost {len(visible_nodes) - len(final_visible)} visible nodes!")
    
    return G_work, final_visible, final_hidden



def visualize_contracted_graph(G_contracted, visible_nodes, hidden_nodes, 
                               use_zephyr_layout=True):
    """
    Visualize the contracted graph showing visible vs hidden nodes.
    """
    # Create x_sol format for compatibility with analyze_maxcut
    x_sol = {}
    for node in G_contracted.nodes():
        x_sol[node] = 1 if node in visible_nodes else 0
    
    # Count cross-edges (visible-hidden connections)
    cross_edges = []
    for u, v in G_contracted.edges():
        if x_sol[u] != x_sol[v]:
            cross_edges.append((u, v))
    
    print(f" Cross-edges (V-H): {len(cross_edges)} out of {G_contracted.number_of_edges()}")
    print(f" Visible nodes: {len(visible_nodes)}")
    print(f" Hidden nodes: {len(hidden_nodes)}")
    
    # Visualize
    if use_zephyr_layout:
        # Use spring layout since Zephyr layout won't work after contraction
        pos = nx.spring_layout(G_contracted, seed=42, k=0.5)
    else:
        pos = nx.spring_layout(G_contracted, seed=42)
    
    node_colors = ['red' if node in visible_nodes else 'blue' 
                   for node in G_contracted.nodes()]
    
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G_contracted, pos, node_color=node_colors, 
                          node_size=50, alpha=0.8)
    nx.draw_networkx_edges(G_contracted, pos, edge_color='lightgray', 
                          alpha=0.3, width=0.5)
    nx.draw_networkx_edges(G_contracted, pos, edgelist=cross_edges, 
                          edge_color='green', alpha=0.6, width=1.0)
    
    # Add legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor='red', markersize=10, label='Visible')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='blue', markersize=10, label='Hidden')
    green_line = plt.Line2D([0], [0], color='green', linewidth=2, 
                            label='V-H edges')
    plt.legend(handles=[red_patch, blue_patch, green_line], loc='upper right')
    
    plt.title(f"Contracted Graph: {len(visible_nodes)} Visible, "
              f"{len(hidden_nodes)} Hidden\n"
              f"{len(cross_edges)} cross-edges")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return x_sol


# Relabel graph nodes to be consecutive: visible first (0 to V-1), then hidden (V to V+H-1)

def relabel_for_training(G_contracted, visible_nodes, hidden_nodes):
    """
    Relabel graph so visible nodes are [0, num_visible-1] 
    and hidden nodes are [num_visible, num_visible+num_hidden-1].
    """
    old_to_new = {}
    
    # Sort visible nodes (handle mixed types)
    try:
        visible_sorted = sorted(visible_nodes)
    except TypeError:
        # Mixed types - convert all to strings for sorting, then use original
        visible_sorted = sorted(visible_nodes, key=str)
    
    # Sort hidden nodes (handle mixed types)
    try:
        hidden_sorted = sorted(hidden_nodes)
    except TypeError:
        hidden_sorted = sorted(hidden_nodes, key=str)
    
    # Visible nodes get IDs 0 to len(visible_nodes)-1
    for new_id, old_id in enumerate(visible_sorted):
        old_to_new[old_id] = new_id
    
    # Hidden nodes get IDs starting from len(visible_nodes)
    offset = len(visible_nodes)
    for new_id, old_id in enumerate(hidden_sorted):
        old_to_new[old_id] = offset + new_id
    
    # Relabel graph
    G_relabeled = nx.relabel_nodes(G_contracted, old_to_new, copy=True)
    
    # New node lists (now all integers)
    visible_new = list(range(len(visible_nodes)))
    hidden_new = list(range(len(visible_nodes), len(visible_nodes) + len(hidden_nodes)))
    
    print(f"Relabeling complete:")
    print(f"  Old node types: {set(type(n).__name__ for n in G_contracted.nodes())}")
    print(f"  New node IDs: all integers [0, {len(G_relabeled.nodes())-1}]")
    
    return G_relabeled, visible_new, hidden_new, old_to_new
