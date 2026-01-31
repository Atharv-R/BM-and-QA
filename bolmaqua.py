'''
This file is a first attempt to create some custom BMs 
based on the D-Wave Zephyr Graph architecture. 

As of 08/12/2025: 
What has been done: 
Mainly: 
- Generate Zephyr graph
- Assign visible & hidden nodes in the graph 
- Train the resulting BM

Also: 
- Removal of some poorly connected nodes in the graph
- Visualizations: Visible vs Hidden in Zephyr graph 

All this is using the previous functionalities from 
custom_bm.py, which implements the custom architecture 
BMs and handles the PCD, sampling, etc 
--> Improvements there make improvements here! 


To do: 
- Need a meaningful assignment of visible nodes to 
pixels in the image. Right now, it's just the order of labels in 
the graph nodes with order of pixel labels. 
    Probably: Group closeby pixels together somehow
- Speed things up (in custom_BMs, probably)
    --> GPU? 
    --> Faster Gibbs sampling (e.g. with Julia?)
    --> Partition the graph for block Gibbs sampling? 
        - Maybe just need to partition Zephyr graph once...
- Hyperparameter tuning (issue: it's slow to run)
    - Right now no images are looking good at all! 
    - Tune things to get good values. 


'''

'''
8/17/25

Tried assigning pixels to nodes in a way that preserves locality.
Instead of first 144 visible, I tried to group nearby pixels together using degrees of each node.

Also, trained using more gibbs step in training (5 -> 10), more epochs (20 -> 50 -> 100), 
'''

'''
8/24/25

Experimented with different assignments for visible. Specifically:
-> assigning second half as visible, and then using the 12x12 grid 
->  removing node relabling after node removal resulted
-> (*the first time I removed node relabling, and assigned second half as visible (no other changes), there was
        weirdly massive improvement in training (PCD ~0), and the samples looked better as well.
         Have been trying to figure out why it happened for just that run, and replicate it for now but no luck :(

8/30/25

Pre-emptive tuning (esp. lower learning rate) to get a decreasing avg PCD loss, which was a success but the quality of samples degraded.
I believe it was an overfitting issue then. 
'''

'''
9/07/25

Integrated downsizing to variable grid_shapes (e.g. 12x12, 13x13, etc.) !

9/15/25
Added a new complete jupyter notebook for Zephyr BMs with variable grid shapes. 
Ran some tests with different grid shapes and K=3,4,5,6 for the Zephyr graph.


9/21/25
Added fusion of hidden nodes for better mixing. Not fully tested yet, but fusion seems to be working to some extent.


9/22/25
Added targeted reduction of hidden nodes 
'''

'''
10/03/25
Changed the logic for fusing hidden nodes. Now, it uses contracted_nodes from networkx to merge low-degree neighboring hidden nodes.
Improvement in training and quality of samples.
'''

from sched import scheduler
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
import gcol
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# --- Config ---
GRID_SHAPE = (12,12)   # <-- change this to (12,12), (13,13), (14,14), etc.
num_visible = GRID_SHAPE[0] * GRID_SHAPE[1]
NUM_VISIBLE = num_visible  # for consistency with other files
grid_shape = GRID_SHAPE  # to be used in other files

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

        G = bm_graph.graph
        coloring_dict = gcol.node_coloring(G)

        # build a numpy array where index i holds coloring_dict[i] (or -1 if missing)
        max_key = max(coloring_dict.keys())
        arr_len = max(G.number_of_nodes(), max_key + 1)
        color_array = np.full(arr_len, -1, dtype=int)
        for k, v in coloring_dict.items():
            color_array[int(k)] = int(v)
        self.coloring = color_array
        #now self.coloring entry i holds the color of node i

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

    def _compute_local_field_v_optimized(self, v_current: torch.Tensor, h_current: torch.Tensor, 
                                        unit_idx: int, W_vv_cols: torch.Tensor, W_vh_rows: torch.Tensor) -> torch.Tensor:
        """GPU-optimized version of local field computation for visible units."""
        # More efficient: avoid clone by computing field without self-connection directly
        field_vv = torch.matmul(v_current, W_vv_cols[:, unit_idx]) - v_current[:, unit_idx] * W_vv_cols[unit_idx, unit_idx]
        field_vh = torch.matmul(h_current, W_vh_rows[unit_idx, :])
        return field_vv + field_vh + self.b_v[unit_idx]

    def _compute_local_field_h(self, v_current: torch.Tensor, h_current: torch.Tensor, unit_idx: int, W_hh: torch.Tensor, W_vh: torch.Tensor) -> torch.Tensor:
        """Computes the local field for a single hidden unit h_j."""
        # Exclude self-connection by masking out the diagonal
        h_masked = h_current.clone()
        h_masked[:, unit_idx] = 0  # Zero out the unit being updated
        
        field_hh = h_masked @ W_hh[:, unit_idx]
        field_hv = v_current @ W_vh[:, unit_idx]
        return field_hh + field_hv + self.b_h[unit_idx]

    def _compute_local_field_h_optimized(self, v_current: torch.Tensor, h_current: torch.Tensor, 
                                        unit_idx: int, W_hh_cols: torch.Tensor, W_vh_direct: torch.Tensor) -> torch.Tensor:
        """GPU-optimized version of local field computation for hidden units."""
        # More efficient: avoid clone by computing field without self-connection directly
        field_hh = torch.matmul(h_current, W_hh_cols[:, unit_idx]) - h_current[:, unit_idx] * W_hh_cols[unit_idx, unit_idx]
        field_hv = torch.matmul(v_current, W_vh_direct[:, unit_idx])
        return field_hh + field_hv + self.b_h[unit_idx]

    @staticmethod
    def _sample_unit_given_field(field: torch.Tensor) -> torch.Tensor:
        """Helper to sample a binary unit given its local field."""
        prob = torch.sigmoid(field)
        return torch.bernoulli(prob)
    
    def free_energy(self, v):
        """
        Approximate free energy for full BM with VV, HH, VH couplings.
        """
        W_vv, W_hh, W_vh = self._get_masked_weights()
        b_v, b_h = self.b_v, self.b_h

        # Visible bias + VV quadratic
        vbias_term = -torch.matmul(v, b_v)
        vv_term = -0.5 * torch.sum(v @ W_vv * v, dim=1)

        # For hidden units: approximate log-partition with mean-field (sigmoid expectation)
        # log( sum_h exp(-E(v,h)) ) ≈ sum_j log(1 + exp(b_h[j] + v @ W_vh[:, j]))
        hidden_term = -torch.sum(
            torch.log1p(torch.exp(b_h + torch.matmul(v, W_vh))), dim=1
        )

        return vbias_term + vv_term + hidden_term

    def gibbs_sample_step(self, v_current: torch.Tensor, h_current: torch.Tensor, 
                          update_v: bool = True, update_h: bool = True, 
                          gibbs_heur_vectorize: bool = False, 
                          track_grad: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one full single-site Gibbs sampling step optimized for GPU.
        
        Args:
            v_current: Current visible state
            h_current: Current hidden state  
            update_v: Whether to update visible units
            update_h: Whether to update hidden units
            gibbs_heur_vectorize: Whether to use vectorized (parallel) sampling
            track_grad: Whether to track gradients. If False, detaches inputs and uses no_grad context.
        
        Returns:
            Tuple of (updated_v, updated_h)
        """
        # Handle gradient tracking and create working copies
        if track_grad:
            v_next = v_current.clone()
            h_next = h_current.clone()
            W_vv, W_hh, W_vh = self._get_masked_weights()
        else:
            # Detach inputs to prevent gradient tracking
            v_next = v_current.detach().clone()
            h_next = h_current.detach().clone()
            
            # Get weights without gradients for sampling
            with torch.no_grad():
                W_vv, W_hh, W_vh = self._get_masked_weights()
                # Detach weights to ensure no gradient tracking
                W_vv = W_vv.detach()
                W_hh = W_hh.detach()
                W_vh = W_vh.detach()

        def _sample_units():
            nonlocal v_next, h_next
            
            # Vectorized version - GPU optimized but heuristic for non-RBM architectures
            if gibbs_heur_vectorize:
                if update_v:
                    # Pre-compute diagonal mask for self-connections (GPU efficient)
                    diag_mask_vv = torch.eye(self.num_visible, device=W_vv.device, dtype=W_vv.dtype)
                    W_vv_masked = W_vv * (1 - diag_mask_vv)  # Zero out diagonal
                    
                    # Vectorized computation: batch_size x num_visible
                    field_vv = torch.matmul(v_next, W_vv_masked)  # V-V interactions (no self-loops)
                    field_vh = torch.matmul(h_next, W_vh.T)  # V-H interactions
                    field_v_all = field_vv + field_vh + self.b_v.unsqueeze(0)  # Add biases
                    
                    # Sample all visible units simultaneously using sigmoid + bernoulli
                    probs_v = torch.sigmoid(field_v_all)
                    v_next.copy_(torch.bernoulli(probs_v))

                if update_h:
                    # Pre-compute diagonal mask for self-connections (GPU efficient)
                    diag_mask_hh = torch.eye(self.num_hidden, device=W_hh.device, dtype=W_hh.dtype)
                    W_hh_masked = W_hh * (1 - diag_mask_hh)  # Zero out diagonal
                    
                    # Vectorized computation: batch_size x num_hidden
                    field_hh = torch.matmul(h_next, W_hh_masked)  # H-H interactions (no self-loops)
                    field_hv = torch.matmul(v_next, W_vh)  # H-V interactions
                    field_h_all = field_hh + field_hv + self.b_h.unsqueeze(0)  # Add biases
                    
                    # Sample all hidden units simultaneously using sigmoid + bernoulli
                    probs_h = torch.sigmoid(field_h_all)
                    h_next.copy_(torch.bernoulli(probs_h))

            # Sequential sampling - proper Gibbs but with GPU optimizations
            else:
                # Pre-generate random permutations on GPU for better performance
                if update_v:
                    v_perm = torch.randperm(self.num_visible, device=v_next.device)
                    # Pre-compute weight slices to avoid repeated indexing
                    W_vv_cols = W_vv.T  # Transpose once for column access
                    W_vh_rows = W_vh  # Keep as is for row access
                    
                    for idx in range(self.num_visible):
                        i = v_perm[idx].item()
                        # More efficient local field computation
                        field_v_i = self._compute_local_field_v_optimized(
                            v_next, h_next, i, W_vv_cols, W_vh_rows)
                        prob = torch.sigmoid(field_v_i)
                        v_next[:, i] = torch.bernoulli(prob)

                if update_h:
                    h_perm = torch.randperm(self.num_hidden, device=h_next.device)
                    # Pre-compute weight slices
                    W_hh_cols = W_hh.T  # Transpose once for column access
                    W_vh_direct = W_vh  # Keep original orientation for hidden field computation
                    
                    for idx in range(self.num_hidden):
                        j = h_perm[idx].item()
                        # More efficient local field computation
                        field_h_j = self._compute_local_field_h_optimized(
                            v_next, h_next, j, W_hh_cols, W_vh_direct)
                        prob = torch.sigmoid(field_h_j)
                        h_next[:, j] = torch.bernoulli(prob)

        # Execute sampling with appropriate gradient context
        if track_grad:
            _sample_units()
        else:
            with torch.no_grad():
                _sample_units()

        return v_next, h_next
    


    def gibbs_sample_step_with_coloring(self, v_current: torch.Tensor, h_current: torch.Tensor, 
                                        update_v=True, update_h=True, track_grad = False, 
                                        gibbs_heur_vectorize = 0):#the heur vectorize is for backwards compatibility
        """
        Performs one full Gibbs sampling step over all units, respecting the graph coloring
        for parallel updates (block Gibbs sampling).
        """
        if v_current.dim() == 1: v_current = v_current.unsqueeze(0)
        if h_current.dim() == 1: h_current = h_current.unsqueeze(0)

        v_state = v_current.clone()
        h_state = h_current.clone()

        if track_grad:
            W_vv, W_hh, W_vh = self._get_masked_weights()
        else:
            # Detach weights to prevent gradient tracking during standard sampling
            with torch.no_grad():
                W_vv, W_hh, W_vh = self._get_masked_weights()
                W_vv, W_hh, W_vh = W_vv.detach(), W_hh.detach(), W_vh.detach()

        def _sample_units_with_coloring(v_state, h_state, W_vv, W_hh, W_vh):
            """
            Performs one full block Gibbs sampling step using the graph coloring.
            Updates all nodes of the same color in parallel.
            """
            num_colors = int(self.coloring.max() + 1)
            
            # These are mappings from the global node index (0 to num_total-1) to the
            # index within the visible or hidden tensors (0 to num_visible-1 or 0 to num_hidden-1).
            v_global_to_local_idx = {node: i for i, node in enumerate(self.bm_graph.visible_nodes)}
            h_global_to_local_idx = {node: i for i, node in enumerate(self.bm_graph.hidden_nodes)}

            for color in range(num_colors):
                # Find global indices for nodes of the current color
                nodes_in_color_global = np.where(self.coloring == color)[0]
                
                # Separate them into visible and hidden
                v_nodes_global = [n for n in nodes_in_color_global if n in self.bm_graph.visible_nodes]
                h_nodes_global = [n for n in nodes_in_color_global if n in self.bm_graph.hidden_nodes]

                # Get the corresponding local indices for tensor slicing
                v_indices_local = [v_global_to_local_idx[n] for n in v_nodes_global]
                h_indices_local = [h_global_to_local_idx[n] for n in h_nodes_global]

                if not v_indices_local and not h_indices_local:
                    continue # No nodes of this color

                # --- Update Visible Units of this color ---
                if v_indices_local and update_v:
                    # Fields from other visible units and all hidden units
                    field_v_from_v = v_state @ W_vv[:, v_indices_local]
                    field_v_from_h = h_state @ W_vh.T[:, v_indices_local]
                    
                    # Total local field for this block of visible units
                    local_field_v = field_v_from_v + field_v_from_h + self.b_v[v_indices_local]
                    
                    # Get probability and sample
                    prob_v = torch.sigmoid(local_field_v)
                    new_v_states = torch.bernoulli(prob_v)
                    
                    # Update the main state tensor
                    v_state[:, v_indices_local] = new_v_states

                # --- Update Hidden Units of this color ---
                if h_indices_local and update_h:
                    # Fields from all visible units and other hidden units
                    field_h_from_v = v_state @ W_vh[:, h_indices_local]
                    field_h_from_h = h_state @ W_hh[:, h_indices_local]

                    # Total local field for this block of hidden units
                    local_field_h = field_h_from_v + field_h_from_h + self.b_h[h_indices_local]
                    
                    # Get probability and sample
                    prob_h = torch.sigmoid(local_field_h)
                    new_h_states = torch.bernoulli(prob_h)
                    
                    # Update the main state tensor
                    h_state[:, h_indices_local] = new_h_states
            
            return v_state, h_state

        v_state, h_state = _sample_units_with_coloring(v_state, h_state, W_vv, W_hh, W_vh)
        
        return v_state, h_state

    # Backward compatibility aliases
    def gibbs_sample_step_no_grad(self, v_current: torch.Tensor, h_current: torch.Tensor, 
                                  update_v: bool = True, update_h: bool = True,
                                  gibbs_heur_vectorize: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DEPRECATED: Use gibbs_sample_step with track_grad=False instead.
        Performs one full single-site Gibbs sampling step WITHOUT gradient tracking.
        """
        return self.gibbs_sample_step(v_current, h_current, update_v, update_h, 
                                    gibbs_heur_vectorize, track_grad=False)

    def mean_field_update(self, v: torch.Tensor, h: torch.Tensor, 
                         update_v: bool = True, update_h: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a mean-field update (deterministic) instead of sampling.
        This can be used for the positive phase to avoid sampling issues.
        """
        W_vv, W_hh, W_vh = self._get_masked_weights()
        v_next = v.clone()
        h_next = h.clone()

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
            #CHANGED this to use "with coloring" gibbs sampling. 
            v_neg, h_neg = self.gibbs_sample_step_with_coloring(v_neg, h_neg, update_v=True, update_h=True, 
                                                        gibbs_heur_vectorize = gibbs_heur_vectorize, 
                                                        track_grad=track_grad)

        # --- Loss Calculation ---
        pos_energy = self.energy(v_pos, h_pos).mean()
        neg_energy = self.energy(v_neg, h_neg).mean()
        cd_loss = pos_energy - neg_energy

        return cd_loss, v_neg

# --------------------------------------------------------------------------



#function to take a graph as input and turn it into a BM with 
# corresponding architecture. 
def graph_to_bm(G, node_labels) -> CustomBoltzmannMachine:
    """
    Factory function that takes a graph and node labels and returns a PyTorch
    Boltzmann Machine model.

    Returns:
        CustomBoltzmannMachine: The initialized PyTorch model.
    """
    print("Building Boltzmann Machine from graph... 🏗️")
    bm_architecture = BoltzmannMachineGraph(G, node_labels)
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
        
# --------------------------------------------------------------------------
# Training: using CD here, though this is relatively flexibly written. 
# def train_boltzmann_machine_cd(model: CustomBoltzmannMachine, data_loader: torch.utils.data.DataLoader,
#                             optimizer: torch.optim.Optimizer, num_epochs: int, k_steps: int,
#                             batch_size: int = 64, step_size: float = 0.001):
#     """
#     Trains the Boltzmann Machine using Contrastive Divergence.
#     batch_size: Number of samples per batch.
#     step_size: Learning rate for optimizer.
#     """
#     loss_history = []
#     model.train()
#     print(f"Starting training on {device} for {num_epochs} epochs... 🏋️")
#     # Update optimizer learning rate if step_size is provided
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = step_size

#     # Re-create data_loader with new batch_size if needed
#     if hasattr(data_loader, 'batch_size') and data_loader.batch_size != batch_size:
#         dataset = data_loader.dataset
#         data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         for batch_data in data_loader:
#             batch = batch_data[0].to(device)
#             optimizer.zero_grad()
#             loss, _ = model(batch, k_steps=k_steps)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         avg_loss = total_loss / len(data_loader)
#         loss_history.append(avg_loss)
#         print(f"Epoch {epoch+1}/{num_epochs} | Avg CD Loss: {avg_loss:.4f}")
#     print("Training complete! ✅")
#     return loss_history

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


def compute_pseudolikelihood(model, v, num_samples=100):
    v = v.to(device)
    batch_size, num_visible = v.shape

    pll_vals = []
    for _ in range(num_samples):
        # Randomly choose which visible unit to flip
        i = torch.randint(0, num_visible, (1,), device=device).item()
        v_flipped = v.clone()
        v_flipped[:, i] = 1 - v_flipped[:, i]

        # Free energies
        fe_orig = model.free_energy(v)
        fe_flip = model.free_energy(v_flipped)

        # Use log-sum-exp trick for stability:
        stacked = torch.stack([-fe_orig, -fe_flip], dim=1)  # shape [batch, 2]
        log_denom = torch.logsumexp(stacked, dim=1)

        # log p(v_i | v_-i)
        log_prob = -fe_orig - log_denom
        pll_vals.append(log_prob.mean().item())

    return np.mean(pll_vals)

# Training with Persistent Contrastive Divergence (PCD) or Contrastive Divergence (CD)
def train_boltzmann_machine_pcd(model: CustomBoltzmannMachine, data_loader: torch.utils.data.DataLoader,
                                optimizer: torch.optim.Optimizer, num_epochs: int, k_steps: int = 1,
                                batch_size: int = 64, step_size: float = 0.001, fused_pairs=None, 
                                track_grad = False, gibbs_heur_vectorize = False, persistent: bool = True):
    """
    Trains the Boltzmann Machine using Persistent Contrastive Divergence (PCD) or standard CD.
    Maintains persistent chains across batches and epochs if persistent=True.
    If persistent=False, it performs standard CD (starts negative chain from data).
    """
    loss_history = []
    pll_values = []
    model.train()
    method_name = "PCD" if persistent else "CD"
    print(f"Starting {method_name} training on {device} for {num_epochs} epochs... 🏋️")

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

            # Initialize or resize persistent chains (only relevant if persistent=True)
            if persistent and ((persistent_v is None) or (persistent_v.shape[0] != current_batch_size)):
                persistent_v = torch.bernoulli(torch.full((current_batch_size, model.num_visible), 0.5, device=device))
                persistent_h = torch.bernoulli(torch.full((current_batch_size, model.num_hidden), 0.5, device=device))

            optimizer.zero_grad()

            # --- Positive Phase (mean-field approx) ---
            v_pos = batch
            h_pos = torch.full((current_batch_size, model.num_hidden), 0.5, device=device)
            for _ in range(model.k_gibbs_positive):
                v_pos, h_pos = model.mean_field_update(v_pos, h_pos, update_v=False, update_h=True)

            # --- Negative Phase ---
            if persistent:
                # PCD: Start from persistent chain
                v_neg = persistent_v.clone()
                h_neg = persistent_h.clone()
            else:
                # CD: Start from data (positive phase result)
                # v_pos is the data batch
                v_neg = v_pos.detach().clone()
                h_neg = h_pos.detach().clone()

            for _ in range(k_steps):
                #FLAG - use grad tracking here? 
                if track_grad: 
                    v_neg, h_neg = model.gibbs_sample_step_with_coloring(v_neg, h_neg, update_v=True, update_h=True,\
                                                           gibbs_heur_vectorize = gibbs_heur_vectorize, 
                                                           track_grad=True)
                else: 
                    v_neg, h_neg = model.gibbs_sample_step_with_coloring(v_neg, h_neg,\
                                                     update_v=True, update_h=True, gibbs_heur_vectorize = gibbs_heur_vectorize, 
                                                     track_grad=False)

            # Update persistent chain if using PCD
            if persistent:
                persistent_v = v_neg.detach()
                persistent_h = h_neg.detach()

            # --- Loss ---
            pos_energy = model.energy(v_pos, h_pos).mean()
            neg_energy = model.energy(v_neg, h_neg).mean()
            # If standard CD, we are minimizing the difference, same loss form:
            pcd_loss = pos_energy - neg_energy

            pcd_loss.backward()

            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        with torch.no_grad():
            pll_val = compute_pseudolikelihood(model, next(iter(data_loader))[0], num_samples=50)
            pll_values.append(pll_val)
            print(f"Epoch {epoch+1}/{num_epochs} | Avg PCD Loss: {avg_loss:.4f} | PLL: {pll_val:.4f}")
            
         if scheduler is not None:
            scheduler.step(pll_val)  # Update LR based on PLL
            current_lr = optimizer.param_groups[0]['lr']
            if epoch > 0 and current_lr != optimizer.param_groups[0]['lr']:
                print(f"  → Learning rate adjusted to {current_lr:.6f}")
    print("PCD training complete! ✅")
    return loss_history, pll_values



#SAMPLING -- this can maybe be improved? ned to test the vectorized version more and think. 
def sample_from_bm(model: CustomBoltzmannMachine, num_samples: int, burn_in_steps: int,
                   method: str = 'gibbs', annealing_schedule: list[float] | None = None, fused_pairs=None,
                   track_grad= False, gibbs_heur_vectorize= False) -> torch.Tensor:
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
                    v, h = model.gibbs_sample_step_with_coloring(v, h, \
                                            track_grad= track_grad, gibbs_heur_vectorize= gibbs_heur_vectorize)
                    # h = enforce_fusion(h, fused_pairs, beta=10.0)
                # if (step + 1) % (burn_in_steps // 5) == 0:  # check periodically
                #         for (i, j) in fused_pairs:
                #             agreement = (h[:, i] == h[:, j]).float().mean().item()
                #             print(f"Fusion check: nodes {i}-{j} agree {agreement*100:.2f}% of the time")
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




def generate_downsized_mnist_torchvision(output_dir=".", grid_shape=GRID_SHAPE, cutoff=0.5):
    """
    Generates downsized MNIST .npy files (from torchvision MNIST, not CSV).
    Resizes 28x28 MNIST images to `grid_shape` (e.g., 12x12, 13x13).
    Saves features and labels as .npy for reuse.
    """
    os.makedirs(output_dir, exist_ok=True)

    def process_dataset(dataset, cutoff):
        downsized, labels = [], []
        for img, label in dataset:
            # torchvision MNIST returns a PIL image unless transform is ToTensor
            img_np = np.array(img, dtype=np.float32)

            # If shape is (28,28,1), squeeze it
            if img_np.ndim == 3:
                img_np = img_np.squeeze(-1)

            img_np = img_np / 255.0  # scale to [0,1]
            img_small = resize(img_np, grid_shape, order=1, anti_aliasing=True, preserve_range=True)

            img_bin = (img_small >= cutoff).astype(np.uint8)
            downsized.append(img_bin.flatten())
            labels.append(label)
        return np.array(downsized), np.array(labels)

    # Load MNIST from torchvision (downloads automatically if not present)
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=None)
    test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=None)

    for split, dataset in [("train", train_data), ("test", test_data)]:
        out_feats = os.path.join(output_dir, f"mnist{grid_shape[0]}x{grid_shape[1]}_{split}feats.npy")
        out_labels = os.path.join(output_dir, f"mnist{grid_shape[0]}x{grid_shape[1]}_{split}labels.npy")
        if not (os.path.exists(out_feats) and os.path.exists(out_labels)):
            print(f"Generating {out_feats} and {out_labels}...")
            feats, labels = process_dataset(dataset, cutoff)
            print("Processed feats shape:", feats.shape) # Debug print
            np.save(out_feats, feats)
            np.save(out_labels, labels)
        else:
            print(f"Found existing {out_feats} and {out_labels}.")

    return (
        np.load(os.path.join(output_dir, f"mnist{grid_shape[0]}x{grid_shape[1]}_trainfeats.npy")),
        np.load(os.path.join(output_dir, f"mnist{grid_shape[0]}x{grid_shape[1]}_trainlabels.npy"))
    )



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
def assign_visibles_by_grid(G, grid_shape=GRID_SHAPE, min_degree=3):
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