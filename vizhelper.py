import dwave_networkx as dnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import resize
import pandas as pd


# Coverts the graph to a Boltzmann Machine and visualizes it.
def visualize_bm_bipartite_layout(G, node_labels, figsize=(10, 6), title="BM Graph - Bipartite Style"):
    import matplotlib.pyplot as plt
    import networkx as nx

    # Split nodes
    visible_nodes = [n for n, t in node_labels.items() if t == 'visible']
    hidden_nodes = [n for n, t in node_labels.items() if t == 'hidden']

    # Manually define bipartite-style positions
    pos = {}

    # Stack visible nodes vertically on the left
    for i, node in enumerate(sorted(visible_nodes)):
        pos[node] = (0, i)

    # Stack hidden nodes vertically on the right
    for i, node in enumerate(sorted(hidden_nodes)):
        pos[node] = (1, i)

    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=visible_nodes, node_color='skyblue', label='Visible', node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='salmon', label='Hidden', node_size=100)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Optionally add labels (can comment out if cluttered)
    # nx.draw_networkx_labels(G, pos, font_size=6)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.legend()
    plt.show()


# def enforce_fusion(h, fused_pairs, beta=10.0):
#     """
#     Enforce strong coupling between fused hidden node pairs at the sampling level.
#     beta -> coupling strength (higher = stronger fusion).
#     """
#     for (i, j) in fused_pairs:
#         # With strong coupling, h[i] and h[j] should agree most of the time.
#         if torch.rand(1).item() < torch.sigmoid(torch.tensor(beta).to(h.device)):
#             # force them equal (take majority or average)
#             val = (h[:, i] + h[:, j]) / 2.0
#             val = (val > 0.5).float()  # binarize to 0/1
#             h[:, i] = val
#             h[:, j] = val
#     return h
''' Code for fusing distribution of nodes by assigning average values to the considered hidden nodes 
'''
# def tie_hidden_parameters(model, fused_pairs):
#     """
#     Enforce parameter tying for fused hidden nodes.
#     Makes their weights and biases identical by averaging.
#     """
#     with torch.no_grad():
#         for (i, j) in fused_pairs:
#             # Tie incoming connections from visible units
#             avg_vh = (model.W_vh_raw[:, i] + model.W_vh_raw[:, j])/2
#             model.W_vh_raw[:, i].copy_(avg_vh)
#             model.W_vh_raw[:, j].copy_(avg_vh)

#             # Tie hidden↔hidden connections (row + col for symmetry)
#             avg_hh_row = (model.W_hh_raw[i, :] + model.W_hh_raw[j, :])/2
#             model.W_hh_raw[i, :].copy_(avg_hh_row)
#             model.W_hh_raw[j, :].copy_(avg_hh_row)

#             avg_hh_col = (model.W_hh_raw[:, i] + model.W_hh_raw[:, j])/2
#             model.W_hh_raw[:, i].copy_(avg_hh_col)
#             model.W_hh_raw[:, j].copy_(avg_hh_col)

#             # Tie hidden biases
#             avg_b = (model.b_h[i] + model.b_h[j]) / 2
#             model.b_h[i].copy_(avg_b)
#             model.b_h[j].copy_(avg_b)

# def check_fusion_strength(model, fused_pairs):
#     with torch.no_grad():
#         for (i, j) in fused_pairs:
#             diff_vh = torch.norm(model.W_vh_raw[:, i] - model.W_vh_raw[:, j]).item()
#             diff_hh_row = torch.norm(model.W_hh_raw[i, :] - model.W_hh_raw[j, :]).item()
#             diff_hh_col = torch.norm(model.W_hh_raw[:, i] - model.W_hh_raw[:, j]).item()
#             diff_b = abs(model.b_h[i] - model.b_h[j]).item()

#             print(f"Fusion check {i}-{j}: "
#                   f"VH {diff_vh:.6f}, HH_row {diff_hh_row:.6f}, HH_col {diff_hh_col:.6f}, Bias {diff_b:.6f}")

def fuse_hidden_nodes(G, hidden_nodes, target_hidden_count):
    """
    Contract hidden nodes until reaching the desired count.
    Always fuses low-degree neighboring hidden nodes.
    """
    hidden_nodes = set(hidden_nodes)

    while len(hidden_nodes) > target_hidden_count:
        # sort hidden nodes by degree
        degrees = [(n, G.degree(n)) for n in hidden_nodes]
        degrees.sort(key=lambda x: x[1])

        fused = False
        for u, _ in degrees:
            # get hidden neighbors of u
            hidden_neighbors = [v for v in G.neighbors(u) if v in hidden_nodes and v != u]
            if not hidden_neighbors:
                continue

            # pick the lowest-degree hidden neighbor
            v = min(hidden_neighbors, key=lambda x: G.degree(x))

            # contract v into u
            G = nx.contracted_nodes(G, u, v, self_loops=False)
            hidden_nodes.remove(v)
            fused = True
            break

        if not fused:
            break

    # relabel nodes consecutively for cleanliness
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    hidden_nodes = [n for n in G.nodes if n >= NUM_VISIBLE]

    return G, hidden_nodes

def get_zephyr_positions(G):
    try:
        # If available, this reflects physical qubit layout
        pos = dnx.zephyr_layout(G)  # dict: node -> (x, y)
    except Exception:
        # fallback: deterministic spring layout
        pos = nx.spring_layout(G, seed=42, dim=2)
    return pos

#  Assigning 144 visibles by laying a 12x12 grid over the layout and
#  greedily taking the nearest unique node to each grid cell center.
def assign_visibles_by_grid(G, grid_shape, min_degree=3):
    pos = get_zephyr_positions(G)
    nodes = np.array(sorted(G.nodes()))
    coords = np.array([pos[n] for n in nodes])  # shape (N, 2)
    if coords.size == 0:
        print("Warning: No eligible nodes found, relaxing min_degree constraint.")
        nodes = list(G.nodes())  # fallback: use all nodes
        coords = np.array([pos[n] for n in nodes])

    # grid centers across the layout bounding box
    xs, ys = coords[:, 0], coords[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    nrows, ncols = grid_shape
    grid_x = np.linspace(xmin, xmax, ncols)
    # flip y so row 0 is top
    grid_y = np.linspace(ymax, ymin, nrows)

    selected = []
    used = set()

    # function to pick nearest unused node to target (x,y), optionally with min degree
    def pick_nearest(target_xy):
        # candidates not used yet
        mask = [n for n in nodes if n not in used]
        if not mask:
            raise RuntimeError("Ran out of nodes to assign as visibles")
        pts = np.array([pos[n] for n in mask])
        d2 = np.sum((pts - target_xy) ** 2, axis=1)
        order = np.argsort(d2)
        if min_degree is None:
            return mask[order[0]]
        # prefer nodes with degree >= min_degree, otherwise next nearest
        for idx in order:
            if G.degree[mask[idx]] >= min_degree:
                return mask[idx]
        return mask[order[0]]  # fallback

    # row-major over the image grid -> preserves pixel locality
    for r in range(nrows):
        for c in range(ncols):
            chosen = pick_nearest(np.array([grid_x[c], grid_y[r]]))
            selected.append(chosen)
            used.add(chosen)

    return selected  # list of node ids in pixel (row-major) order

#   Relabel graph so selected visibles become 0..(v-1) in pixel order.
#    All remaining nodes (hidden) become v..(v+h-1).
def relabel_visible_first(G, visible_nodes_in_pixel_order):
    visible_set = set(visible_nodes_in_pixel_order)
    hidden_nodes = [n for n in sorted(G.nodes()) if n not in visible_set]

    mapping = {}
    # visibles first in the order that matches pixel order
    for new_i, old in enumerate(visible_nodes_in_pixel_order):
        mapping[old] = new_i
    # hiddens after
    offset = len(visible_nodes_in_pixel_order)
    for j, old in enumerate(hidden_nodes):
        mapping[old] = offset + j

    G2 = nx.relabel_nodes(G, mapping, copy=True)
    return G2, mapping

def draw_zephyr_hidden_visible(G, vh_nodearray):
    '''
    Draws G as a zephyr graph topology, with visible / hidden nodes as labeled in vh_nodearray
    G: Networkx Graph
    vh_nodearray: numpy array of 0 and 1s -- 1 for visible, 0 for hidden

    hidden nodes will be blue, and visibles red. 
    '''
    vh_dict = {} #dnx.draw needs dict 

    #assigning large negative or positive value for v vs h
    for i, entry in enumerate(vh_nodearray):
        vh_dict[i] = (entry-0.5)*1000

    #using linear biases, which dnx can color code, 
    # to do the actual coloring 
    dnx.draw_zephyr(G, linear_biases=vh_dict)
    return None

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