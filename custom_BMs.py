'''
This is a first attempt at training custom architecture BMs, 
and includes some hopefully helpful functions for the future. 
This file can (and should) be used as a module in other files in this folder.

(e.g. probably better to remove this main() here if things work 
as intended, the main() at the bottom is just for testing purposes)
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


#this is a class to use for the networkx graph to boltzmann machine conversion.
class BoltzmannMachineGraph:
    """
    Defines the Boltzmann Machine architecture based on a NetworkX graph
    and a mapping of nodes to visible/hidden units. Generates masks for weight matrices.
    """
    def __init__(self, graph: nx.Graph, node_labels):
        """
        Args:
            graph (nx.Graph): The undirected graph defining connections.
            node_labels (dict or list): A dictionary mapping node IDs to 'visible' or 'hidden'.
        """
        if not (isinstance(node_labels, dict) or isinstance(node_labels, np.ndarray)):
            raise TypeError("node_labels must be either a dict or a numpy array.")
        if isinstance(node_labels, np.ndarray):
            # Convert numpy array to dict: assume 0=visible, 1=hidden or vice versa
            node_labels = {i: 'visible' if label == 1 else 'hidden' for i, label in enumerate(node_labels)}
        self.graph = graph
        self.node_labels = node_labels

        if not all(label in ['visible', 'hidden'] for label in node_labels.values()):
            raise ValueError("Node labels must be 'visible' or 'hidden'.")
        if set(graph.nodes()) != set(node_labels.keys()):
            raise ValueError("All graph nodes must have a label in node_labels.")

        self.visible_nodes = sorted([node for node, label in node_labels.items() if label == 'visible'])
        self.hidden_nodes = sorted([node for node, label in node_labels.items() if label == 'hidden'])

        self.num_visible = len(self.visible_nodes)
        self.num_hidden = len(self.hidden_nodes)

        self.v_node_to_idx = {node: i for i, node in enumerate(self.visible_nodes)}
        self.h_node_to_idx = {node: i for i, node in enumerate(self.hidden_nodes)}

        self.mask_vv, self.mask_hh, self.mask_vh = self._generate_adjacency_masks()

    def _generate_adjacency_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates boolean masks for W_vv, W_hh, and W_vh based on the graph."""
        mask_vv = torch.zeros(self.num_visible, self.num_visible, dtype=torch.bool, device=device)
        mask_hh = torch.zeros(self.num_hidden, self.num_hidden, dtype=torch.bool, device=device)
        mask_vh = torch.zeros(self.num_visible, self.num_hidden, dtype=torch.bool, device=device)

        for u, v in self.graph.edges():
            u_label, v_label = self.node_labels[u], self.node_labels[v]

            if u_label == 'visible' and v_label == 'visible':
                u_idx, v_idx = self.v_node_to_idx[u], self.v_node_to_idx[v]
                mask_vv[u_idx, v_idx] = True
                mask_vv[v_idx, u_idx] = True # Ensure symmetry for undirected graph

            elif u_label == 'hidden' and v_label == 'hidden':
                u_idx, v_idx = self.h_node_to_idx[u], self.h_node_to_idx[v]
                mask_hh[u_idx, v_idx] = True
                mask_hh[v_idx, u_idx] = True # Ensure symmetry for undirected graph

            else: # visible-hidden connection
                if u_label == 'visible':
                    u_idx, v_idx = self.v_node_to_idx[u], self.h_node_to_idx[v]
                else: # u_label is 'hidden'
                    u_idx, v_idx = self.v_node_to_idx[v], self.h_node_to_idx[u]
                mask_vh[u_idx, v_idx] = True

        mask_vv.fill_diagonal_(False)
        mask_hh.fill_diagonal_(False)

        return mask_vv, mask_hh, mask_vh


#this is a custom BM model through PyTorch, which defines 
# a lot of the relevant functions as methods in this class. 
# (e.g. energy, sampling, etc.)
class CustomBoltzmannMachine(nn.Module):
    """
    A PyTorch Boltzmann Machine with a custom architecture defined by masks.
    Supports binary units {0, 1}.
    """
    def __init__(self, bm_graph: BoltzmannMachineGraph, k_gibbs_positive: int = 3):
        super().__init__()
        self.bm_graph = bm_graph
        self.num_visible = bm_graph.num_visible
        self.num_hidden = bm_graph.num_hidden
        self.k_gibbs_positive = k_gibbs_positive

        self.register_buffer('mask_vv', bm_graph.mask_vv)
        self.register_buffer('mask_hh', bm_graph.mask_hh)
        self.register_buffer('mask_vh', bm_graph.mask_vh)

        self.W_vv_raw = nn.Parameter(torch.randn(self.num_visible, self.num_visible, device=device) * 0.01)
        self.W_hh_raw = nn.Parameter(torch.randn(self.num_hidden, self.num_hidden, device=device) * 0.01)
        self.W_vh_raw = nn.Parameter(torch.randn(self.num_visible, self.num_hidden, device=device) * 0.01)

        self.b_v = nn.Parameter(torch.zeros(self.num_visible, device=device))
        self.b_h = nn.Parameter(torch.zeros(self.num_hidden, device=device))

    def _get_masked_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies masks and enforces symmetry for intra-layer weights."""
        W_vv = (self.W_vv_raw + self.W_vv_raw.T) / 2 * self.mask_vv
        W_hh = (self.W_hh_raw + self.W_hh_raw.T) / 2 * self.mask_hh
        W_vh = self.W_vh_raw * self.mask_vh
        return W_vv, W_hh, W_vh

    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the energy of a given state (v, h).
        E(v,h) = -0.5*v'W_vv*v - 0.5*h'W_hh*h - v'W_vh*h - b_v'*v - b_h'*h
        """
        W_vv, W_hh, W_vh = self._get_masked_weights()
        if v.dim() == 1: v = v.unsqueeze(0)
        if h.dim() == 1: h = h.unsqueeze(0)

        term_vv = -0.5 * torch.sum(v @ W_vv * v, dim=1)
        term_hh = -0.5 * torch.sum(h @ W_hh * h, dim=1)
        term_vh = -torch.sum((v @ W_vh) * h, dim=1)
        term_bv = -v @ self.b_v
        term_bh = -h @ self.b_h

        return term_vv + term_hh + term_vh + term_bv + term_bh

    def _compute_local_field_v(self, v_current: torch.Tensor, h_current: torch.Tensor, unit_idx: int, W_vv: torch.Tensor, W_vh: torch.Tensor) -> torch.Tensor:
        """Computes the local field for a single visible unit v_i."""
        # Exclude self-connection by masking out the diagonal
        v_masked = v_current.clone()
        v_masked[:, unit_idx] = 0  # Zero out the unit being updated
        
        field_vv = v_masked @ W_vv[:, unit_idx]
        field_vh = h_current @ W_vh[unit_idx, :].T
        return field_vv + field_vh + self.b_v[unit_idx]

    def _compute_local_field_h(self, v_current: torch.Tensor, h_current: torch.Tensor, unit_idx: int, W_hh: torch.Tensor, W_vh: torch.Tensor) -> torch.Tensor:
        """Computes the local field for a single hidden unit h_j."""
        # Exclude self-connection by masking out the diagonal
        h_masked = h_current.clone()
        h_masked[:, unit_idx] = 0  # Zero out the unit being updated
        
        field_hh = h_masked @ W_hh[:, unit_idx]
        field_hv = v_current @ W_vh[:, unit_idx]
        return field_hh + field_hv + self.b_h[unit_idx]

    @staticmethod
    def _sample_unit_given_field(field: torch.Tensor) -> torch.Tensor:
        """Helper to sample a binary unit given its local field."""
        prob = torch.sigmoid(field)
        return torch.bernoulli(prob)

    def gibbs_sample_step_no_grad(self, v_current: torch.Tensor, h_current: torch.Tensor, 
                                  update_v: bool = True, update_h: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one full single-site Gibbs sampling step WITHOUT gradient tracking.
        This version is used for sampling during training to avoid in-place operation errors.
        """
        # Detach inputs to prevent gradient tracking
        v_next = v_current.detach().clone()
        h_next = h_current.detach().clone()
        
        # Get weights without gradients for sampling
        with torch.no_grad():
            W_vv, W_hh, W_vh = self._get_masked_weights()
            W_vv = W_vv.detach()
            W_hh = W_hh.detach()
            W_vh = W_vh.detach()


            # # vectorized, but not sure if this is sound mathematically but fast!
            # if update_v:
            #     # Vectorized update for all visible units
            #     fields_v = torch.stack([
            #         self._compute_local_field_v(v_next, h_next, i, W_vv, W_vh)
            #         for i in range(self.num_visible)
            #     ], dim=1)
            #     v_next = self._sample_unit_given_field(fields_v)

            # if update_h:
            #     # Vectorized update for all hidden units
            #     fields_h = torch.stack([
            #         self._compute_local_field_h(v_next, h_next, j, W_hh, W_vh)
            #         for j in range(self.num_hidden)
            #     ], dim=1)
            #     h_next = self._sample_unit_given_field(fields_h)

            #OLD method, not vectorized!
            if update_v:
                    # Sequentially update each visible unit
                    for i in torch.randperm(self.num_visible):
                        field_v_i = self._compute_local_field_v(v_next, h_next, i, W_vv, W_vh)
                        v_next[:, i] = self._sample_unit_given_field(field_v_i)

            if update_h:
                # Sequentially update each hidden unit
                for j in torch.randperm(self.num_hidden):
                    field_h_j = self._compute_local_field_h(v_next, h_next, j, W_hh, W_vh)
                    h_next[:, j] = self._sample_unit_given_field(field_h_j)

        return v_next, h_next

    def gibbs_sample_step(self, v_current: torch.Tensor, h_current: torch.Tensor, 
                          update_v: bool = True, update_h: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one full single-site Gibbs sampling step WITH gradient tracking.
        This version is used for inference/generation after training.
        """
        v_next = v_current.clone()
        h_next = h_current.clone()
        
        W_vv, W_hh, W_vh = self._get_masked_weights()

        if update_v:
            # Sequentially update each visible unit
            for i in torch.randperm(self.num_visible):
                field_v_i = self._compute_local_field_v(v_next, h_next, i, W_vv, W_vh)
                v_next[:, i] = self._sample_unit_given_field(field_v_i)

        if update_h:
            # Sequentially update each hidden unit
            for j in torch.randperm(self.num_hidden):
                field_h_j = self._compute_local_field_h(v_next, h_next, j, W_hh, W_vh)
                h_next[:, j] = self._sample_unit_given_field(field_h_j)

        return v_next, h_next

    def mean_field_update(self, v: torch.Tensor, h: torch.Tensor, 
                         update_v: bool = True, update_h: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a mean-field update (deterministic) instead of sampling.
        This can be used for the positive phase to avoid sampling issues.
        """
        W_vv, W_hh, W_vh = self._get_masked_weights()
        v_next = v.clone()
        h_next = h.clone()
        
        # # Vectorized (not sure if this is sound mathematically but fast!)
        # if update_v:
        #     # Vectorized update for all visible units
        #     fields_v = torch.stack([
        #     self._compute_local_field_v(v_next, h_next, i, W_vv, W_vh)
        #     for i in range(self.num_visible)
        #     ], dim=1)
        #     v_next = torch.sigmoid(fields_v)

        # if update_h:
        #     # Vectorized update for all hidden units
        #     fields_h = torch.stack([
        #     self._compute_local_field_h(v_next, h_next, j, W_hh, W_vh)
        #     for j in range(self.num_hidden)
        #     ], dim=1)
        #     h_next = torch.sigmoid(fields_h)

        # # old, not vectorized! 
        if update_v:
            for i in range(self.num_visible):
                field_v_i = self._compute_local_field_v(v_next, h_next, i, W_vv, W_vh)
                v_next[:, i] = torch.sigmoid(field_v_i)
        
        if update_h:
            for j in range(self.num_hidden):
                field_h_j = self._compute_local_field_h(v_next, h_next, j, W_hh, W_vh)
                h_next[:, j] = torch.sigmoid(field_h_j)
                
        return v_next, h_next

    def forward(self, v_data: torch.Tensor, k_steps: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the Contrastive Divergence (CD-k) update.
        Uses mean-field for positive phase and sampling for negative phase.
        """
        batch_size = v_data.shape[0]
        v_pos = v_data.to(device)

        # --- Positive Phase: Use mean-field approximation ---
        # Initialize hidden units
        h_pos = torch.full((batch_size, self.num_hidden), 0.5, device=device)
        
        # Run mean-field updates to get approximate posterior
        for _ in range(self.k_gibbs_positive):
            v_pos, h_pos = self.mean_field_update(v_pos, h_pos, update_v=False, update_h=True)

        # --- Negative Phase: Use sampling ---
        # Start from positive phase but detached
        v_neg = v_pos.detach().clone()
        h_neg = h_pos.detach().clone()
        
        # Sample from the model
        for _ in range(k_steps):
            v_neg, h_neg = self.gibbs_sample_step_no_grad(v_neg, h_neg, update_v=True, update_h=True)

        # --- Loss Calculation ---
        pos_energy = self.energy(v_pos, h_pos).mean()
        neg_energy = self.energy(v_neg, h_neg).mean()
        cd_loss = pos_energy - neg_energy

        return cd_loss, v_neg

# --------------------------------------------------------------------------
def build_custom_bm_graph(num_visible: int,
                          num_hidden: int,
                          visible_visible_edges: list[tuple[int, int]],
                          hidden_hidden_edges: list[tuple[int, int]],
                          visible_hidden_edges: list[tuple[int, int]]) -> tuple[nx.Graph, dict[int, str]]:
    """
    Constructs a NetworkX graph for your Boltzmann Machine given explicit edge lists.

    Nodes are assumed to be indexed as:
        0 .. num_visible-1           : visible units
        num_visible .. num_visible+num_hidden-1 : hidden units

    Edge lists should use those indices. Example:
        visible_hidden_edges = [(0, 144), (3, 150), ...]
    """
    num_nodes = num_visible + num_hidden
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Add edges; assume inputs are valid (could add sanity checks)
    G.add_edges_from(visible_visible_edges)
    G.add_edges_from(hidden_hidden_edges)
    G.add_edges_from(visible_hidden_edges)

    # Labeling
    node_labels = {i: 'visible' if i < num_visible else 'hidden' for i in range(num_nodes)}
    return G, node_labels

# visualizing the graph of the BM
def visualize_bm_graph(G: nx.Graph, node_labels: dict[int, str],
                       figsize=(8, 6), title="Boltzmann Machine Graph",
                       show_edge_weights: bool = False):
    """
    Draws the BM graph with visible/hidden nodes colored differently.
    """
    plt.figure(figsize=figsize)
    # layout can be changed: spring_layout, shell_layout, spectral_layout, etc.
    pos = nx.spring_layout(G, seed=42)  # deterministic for consistency

    # Partition nodes
    visible_nodes = [n for n, t in node_labels.items() if t == 'visible']
    hidden_nodes = [n for n, t in node_labels.items() if t == 'hidden']

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=visible_nodes, node_color='skyblue', label='Visible', node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='salmon', label='Hidden', node_size=100)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Optionally annotate edge weights if present
    if show_edge_weights:
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            w = data.get('weight', '')
            edge_labels[(u, v)] = f"{w:.2f}" if isinstance(w, float) else str(w)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Labels (optional - can clutter if many nodes)
    # nx.draw_networkx_labels(G, pos, font_size=6)

    plt.title(title)
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#function to take a graph as input and turn it into a BM with 
# corresponding architecture. 
def graph_to_bm(graph: nx.Graph, node_labels: dict[int, str]) -> CustomBoltzmannMachine:
    """
    Factory function that takes a graph and node labels and returns a PyTorch
    Boltzmann Machine model.

    Args:
        graph (nx.Graph): The NetworkX graph defining the model's architecture.
        node_labels (dict): A dictionary mapping node IDs to 'visible' or 'hidden'.

    Returns:
        CustomBoltzmannMachine: The initialized PyTorch model.
    """
    print("Building Boltzmann Machine from graph... 🏗️")
    bm_architecture = BoltzmannMachineGraph(graph, node_labels)
    model = CustomBoltzmannMachine(bm_architecture).to(device)
    return model

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

# --------------------------------------------------------------------------
# Training: using CD here, though this is relatively flexibly written. 
def train_boltzmann_machine_cd(model: CustomBoltzmannMachine, data_loader: torch.utils.data.DataLoader,
                            optimizer: torch.optim.Optimizer, num_epochs: int, k_steps: int,
                            batch_size: int = 64, step_size: float = 0.001):
    """
    Trains the Boltzmann Machine using Contrastive Divergence.
    batch_size: Number of samples per batch.
    step_size: Learning rate for optimizer.
    """
    loss_history = []
    model.train()
    print(f"Starting training on {device} for {num_epochs} epochs... 🏋️")
    # Update optimizer learning rate if step_size is provided
    for param_group in optimizer.param_groups:
        param_group['lr'] = step_size

    # Re-create data_loader with new batch_size if needed
    if hasattr(data_loader, 'batch_size') and data_loader.batch_size != batch_size:
        dataset = data_loader.dataset
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_data in data_loader:
            batch = batch_data[0].to(device)
            optimizer.zero_grad()
            loss, _ = model(batch, k_steps=k_steps)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg CD Loss: {avg_loss:.4f}")
    print("Training complete! ✅")
    return loss_history

# Training with Persistent Contrastive Divergence (PCD)
def train_boltzmann_machine_pcd(model: CustomBoltzmannMachine, data_loader: torch.utils.data.DataLoader,
                                optimizer: torch.optim.Optimizer, num_epochs: int, k_steps: int = 1,
                                batch_size: int = 64, step_size: float = 0.001):
    """
    Trains the Boltzmann Machine using Persistent Contrastive Divergence (PCD).
    Maintains persistent chains across batches and epochs.
    """
    loss_history = []
    model.train()
    print(f"Starting PCD training on {device} for {num_epochs} epochs... 🏋️")

    for param_group in optimizer.param_groups:
        param_group['lr'] = step_size

    if hasattr(data_loader, 'batch_size') and data_loader.batch_size != batch_size:
        dataset = data_loader.dataset
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Persistent chains (initialized during first batch)
    persistent_v = None
    persistent_h = None

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_data in data_loader:
            batch = batch_data[0].to(device)
            current_batch_size = batch.shape[0]

            # Initialize or resize persistent chains
            if (persistent_v is None) or (persistent_v.shape[0] != current_batch_size):
                persistent_v = torch.bernoulli(torch.full((current_batch_size, model.num_visible), 0.5, device=device))
                persistent_h = torch.bernoulli(torch.full((current_batch_size, model.num_hidden), 0.5, device=device))

            optimizer.zero_grad()

            # --- Positive Phase (mean-field approx) ---
            v_pos = batch
            h_pos = torch.full((current_batch_size, model.num_hidden), 0.5, device=device)
            for _ in range(model.k_gibbs_positive):
                v_pos, h_pos = model.mean_field_update(v_pos, h_pos, update_v=False, update_h=True)

            # --- Negative Phase (persistent chain) ---
            v_neg = persistent_v.clone()
            h_neg = persistent_h.clone()
            for _ in range(k_steps):
                v_neg, h_neg = model.gibbs_sample_step_no_grad(v_neg, h_neg, update_v=True, update_h=True)

            # Update persistent chain
            persistent_v = v_neg.detach()
            persistent_h = h_neg.detach()

            # --- Loss ---
            pos_energy = model.energy(v_pos, h_pos).mean()
            neg_energy = model.energy(v_neg, h_neg).mean()
            pcd_loss = pos_energy - neg_energy

            pcd_loss.backward()
            optimizer.step()
            total_loss += pcd_loss.item()

        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg PCD Loss: {avg_loss:.4f}")
    print("PCD training complete! ✅")
    return loss_history



#SAMPLING -- this can maybe be improved? ned to test the vectorized version more and think. 
def sample_from_bm(model: CustomBoltzmannMachine, num_samples: int, burn_in_steps: int,
                   method: str = 'gibbs', annealing_schedule: list[float] | None = None) -> torch.Tensor:
    """
    Samples visible unit configurations from the trained Boltzmann Machine.
    """
    model.eval()
    
    samples = []
    if method == 'gibbs':
        print(f"Running Gibbs sampler for {num_samples} samples, {burn_in_steps} burn-in steps each... 🔥")
        with torch.no_grad():
            for n in range(num_samples):
                v = torch.bernoulli(torch.full((1, model.num_visible), 0.5, device=device))
                h = torch.bernoulli(torch.full((1, model.num_hidden), 0.5, device=device))
                for step in range(burn_in_steps):
                    v, h = model.gibbs_sample_step(v, h)
                samples.append(v.squeeze(0).clone())
                if (n + 1) % (max(1, num_samples // 10)) == 0:
                    print(f"  Gibbs sample {n+1}/{num_samples}")
        print("Gibbs sampling complete.")
        return torch.stack(samples, dim=0)

    elif method == 'simulated_annealing':
        print(f"Running Simulated Annealing for {num_samples} samples, {burn_in_steps} steps each... ❄️")
        if annealing_schedule is None:
            annealing_schedule = np.logspace(np.log10(10.0), np.log10(1), burn_in_steps)

        if len(annealing_schedule) != burn_in_steps:
            raise ValueError("Length of annealing_schedule must match burn_in_steps.")

        W_vv, W_hh, W_vh = model._get_masked_weights()
        W_vv = W_vv.detach()
        W_hh = W_hh.detach()
        W_vh = W_vh.detach()

        with torch.no_grad():
            for n in range(num_samples):
                v = torch.bernoulli(torch.full((1, model.num_visible), 0.5, device=device))
                h = torch.bernoulli(torch.full((1, model.num_hidden), 0.5, device=device))
                for step, temp in enumerate(annealing_schedule):
                    # Visible units
                    for i in torch.randperm(model.num_visible):
                        field_v = model._compute_local_field_v(v, h, i, W_vv, W_vh)
                        delta_E = field_v * (1.0 - 2.0 * v[:, i])
                        accept_prob = torch.exp(-delta_E / temp).clamp(0, 1)
                        flip_mask = torch.bernoulli(accept_prob).bool()
                        v[flip_mask, i] = 1.0 - v[flip_mask, i]

                    # Hidden units
                    for j in torch.randperm(model.num_hidden):
                        field_h = model._compute_local_field_h(v, h, j, W_hh, W_vh)
                        delta_E = field_h * (1.0 - 2.0 * h[:, j])
                        accept_prob = torch.exp(-delta_E / temp).clamp(0, 1)
                        flip_mask = torch.bernoulli(accept_prob).bool()
                        h[flip_mask, j] = 1.0 - h[flip_mask, j]
                samples.append(v.squeeze(0).clone())
                if (n + 1) % (max(1, num_samples // 10)) == 0:
                    print(f"  SA sample {n+1}/{num_samples}")
        print("Simulated Annealing complete.")
        return torch.stack(samples, dim=0)
    else:
        raise ValueError("Method must be 'gibbs' or 'simulated_annealing'.")


# --- Tabu Search to Improve Sample Likelihood ---
def tabu_search_bm(model, v_init, h_init=None, steps=5, tabu_size=10):
    """
    Performs Tabu Search to locally improve the likelihood of a visible sample under the BM.
    Only flips one bit at a time, keeps a tabu list of recent states.
    This is to make the produced samples post-training nicer. 
    """
    v = v_init.clone().detach()
    if h_init is None:
        # Use mean-field to initialize hidden units
        h = torch.full((1, model.num_hidden), 0.5, device=device)
        for _ in range(model.k_gibbs_positive):
            _, h = model.mean_field_update(v, h, update_v=False, update_h=True)
    else:
        h = h_init.clone().detach()

    tabu_list = []
    best_v = v.clone()
    best_h = h.clone()
    best_energy = model.energy(v, h).item()

    for step in range(steps):
        candidates = []
        energies = []
        for i in range(model.num_visible):
            v_candidate = v.clone()
            v_candidate[0, i] = 1.0 - v_candidate[0, i]  # Flip bit
            # Check if candidate is in tabu list
            if any(torch.equal(v_candidate, t) for t in tabu_list):
                continue
            # Update hidden units using mean-field
            h_candidate = h.clone()
            _, h_candidate = model.mean_field_update(v_candidate, h_candidate, update_v=False, update_h=True)
            energy = model.energy(v_candidate, h_candidate).item()
            candidates.append((v_candidate, h_candidate))
            energies.append(energy)
        if not candidates:
            break
        min_idx = int(np.argmin(energies))
        v, h = candidates[min_idx]
        tabu_list.append(v.clone())
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        if energies[min_idx] < best_energy:
            best_v = v.clone()
            best_h = h.clone()
            best_energy = energies[min_idx]
    return best_v.squeeze(0)

# --------------------------------------------------------------------------
## 5. Main Execution
# --------------------------------------------------------------------------

def main():
    # --- Dataset Loading ---
    print("Loading dataset...")
    try:
        mnist_feats = np.load('mnist12x12_trainfeats.npy')
        mnist_labels = np.load('mnist12x12_trainlabels.npy')
    except FileNotFoundError:
        print("\n Error: 'mnist12x12_trainfeats.npy' not found.")
        print("Please ensure the dataset is available or create synthetic data for testing.")
        # Create synthetic binary data for testing
        print("Creating synthetic binary data for testing...")
        mnist_feats = np.random.rand(1000, 144) > 0.3  # 1000 samples, 144 features
        mnist_feats = mnist_feats.astype(np.float32)
        mnist_labels = mnist_labels.astype(np.float32)

    # Only keep images with label 0 or 1
    mnist_feats = mnist_feats[(mnist_labels == 0) | (mnist_labels == 1)]
    mnist_labels = mnist_labels[(mnist_labels == 0) | (mnist_labels == 1)]

    #cut size of dataset massively for faster training
    mnist_feats = mnist_feats[:200]  # Use only the first few images for quick testing
    mnist_feats = (mnist_feats - mnist_feats.min()) / (mnist_feats.max() - mnist_feats.min())
    # Binarize and create DataLoader
    roundup_boost = 0
    X_data = (torch.from_numpy(mnist_feats).float() < 0.5 + roundup_boost).float()
    dataset = torch.utils.data.TensorDataset(X_data)

    # --- Visualize a few images from X_data ---
    plt.figure(figsize=(8, 2))
    for i in range(8):
        plt.subplot(1, 8, i + 1)
        plt.imshow(X_data[i].cpu().numpy().reshape(12, 12), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    plt.suptitle("Example images from X_data")
    plt.tight_layout()
    plt.show()
    

    # Custom Graph setup
    num_visible = 144
    num_hidden = 50

    # Suppose we have these edge definitions:
    vv = [(0, 1), (2, 3)]  # visible-visible
    hh = [(144, 145), (146, 147)]  # hidden-hidden (indices start at num_visible)
    vh = [(0, 144), (5, 149), (10, 160)]  # visible-hidden (make sure hidden indices are offset)

    G, node_labels = build_custom_bm_graph(num_visible, num_hidden, vv, hh, vh)

    
    # --- Graph and Model Setup ---
    # num_visible = 144  # 12x12 pixels
    # num_hidden = 50    # Reduced for faster training
    # num_nodes = num_visible + num_hidden
    # er_p = 0.4  # Further reduced connection probability for stability

    step_size = 0.01  # You can change this value for experiments 
    l2_amount = 0.1  # L2 regularization amount
    num_epochs = 50
    batch_size = 30  
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # make_bipartite = False  # toggle to True for RBM

    # print(f"Generating Erdos-Renyi graph (n={num_nodes}, p={er_p})...")
    # G = nx.erdos_renyi_graph(num_nodes, er_p, seed=42)
    # node_labels = {i: 'visible' if i < num_visible else 'hidden' for i in range(num_nodes)}
    
    # if make_bipartite:
    #     # Remove all edges that are not between visible and hidden nodes
    #     edges_to_remove = [(u, v) for u, v in G.edges() if (node_labels[u] == node_labels[v])]
    #     G.remove_edges_from(edges_to_remove)

    # Print some graph statistics
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    vv_edges = sum(1 for u, v in G.edges() if u < num_visible and v < num_visible)
    hh_edges = sum(1 for u, v in G.edges() if u >= num_visible and v >= num_visible)
    vh_edges = sum(1 for u, v in G.edges() if (u < num_visible) != (v < num_visible))
    print(f"V-V edges: {vv_edges}, H-H edges: {hh_edges}, V-H edges: {vh_edges}")

    visualize_bm_graph(G, node_labels, title="Custom BM Architecture")
    visualize_bm_bipartite_layout(G, node_labels, title="My Custom BM Layout (Bipartite Style)")


    # Use the factory function
    # model_cd = graph_to_bm(G, node_labels)
    model_pcd = graph_to_bm(G, node_labels)

    # --- Training ---
    # optimizer_cd = torch.optim.RMSprop(model_cd.parameters(), lr=step_size, weight_decay=l2_amount)
    optimizer_pcd = torch.optim.RMSprop(model_pcd.parameters(), lr=step_size, weight_decay=l2_amount)
    # cd_losses = train_boltzmann_machine_cd(model_cd, data_loader, optimizer_cd, num_epochs=num_epochs, k_steps=1, batch_size=batch_size, step_size=step_size)
    pcd_losses = train_boltzmann_machine_pcd(model_pcd, data_loader, optimizer_pcd, num_epochs=num_epochs, k_steps=1, batch_size=batch_size, step_size=step_size)
    
    # --- Plot training loss ---
    # plt.plot(cd_losses, label='CD')
    plt.plot(pcd_losses, label='PCD')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Energy Loss')
    plt.title('Training Loss Comparison: CD vs PCD')
    plt.legend()
    plt.grid(True)
    plt.show()

    #--- Sampling & Visualization ---
    num_gen_samples = 16
    burn_in = 500
    

    #need to find a way to speed up these sampling functions, so slow
    print("\n--- Generating samples using Gibbs Sampling ---")
    #gibbs_samples = sample_from_bm(model, num_gen_samples, burn_in, method='gibbs')
    # samples_cd = sample_from_bm(model_cd, num_samples=64, burn_in_steps=100)
    samples_pcd = sample_from_bm(model_pcd, num_gen_samples, burn_in, method='gibbs')

    #print("\n--- Generating samples using Simulated Annealing ---")
    #sa_samples = sample_from_bm(model, num_gen_samples, burn_in, method='simulated_annealing')
    
    def plot_samples(samples: torch.Tensor, title: str):
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < samples.shape[0]:
                ax.imshow(samples[i].cpu().numpy().reshape(12, 12), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    print("\nDisplaying generated images... 🖼️")
    # plot_samples(samples_cd, "Samples from Gibbs Sampling (CD)")
    plot_samples(samples_pcd, "Samples from Gibbs Sampling (PCD)")
    #plot_samples(sa_samples, "Samples from Simulated Annealing")


    print("\n--- Improving samples with Tabu Search ---")
    tabu_steps = 5
    tabu_improved_samples = []
    for i in range(samples_pcd.shape[0]):
        v_init = samples_pcd[i].unsqueeze(0)
        improved_v = tabu_search_bm(model_pcd, v_init, steps=tabu_steps)
        tabu_improved_samples.append(improved_v)
    tabu_improved_samples = torch.stack(tabu_improved_samples, dim=0)

    print("\nDisplaying Tabu Search improved images... 🖼️")
    plot_samples(tabu_improved_samples, f"Tabu Search Improved Samples ({tabu_steps} steps)")


if __name__ == '__main__':
    main()
# %%
