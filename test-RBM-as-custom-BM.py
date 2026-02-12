'''
This tests our custom BM code implementation using an RBM-style bipartite graph.
We know that RBMs can definitely learn the MNIST data, 
and that they can be trained to produce good samples, 
so this is a good check to see if everything is working. 

As long as this is NOT working, 
there is some issue with some part of our implementation! 
Points to check in on would be training algorotihms, sampling methods,
and hyperparameter choices.

Test Jan 22, 2026: 
with 

lr = 0.005
batch_size = 64
epochs = 10
k_steps = 5  # CD-k / PCD-k steps

the samples generated look decent, 
but they can definitely be improved! 

Let's do some testing and record data of what works best. 
'''
#%% defs


import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Import from bolmaqua
# We assume bolmaqua.py is in the same directory and contains the necessary classes and functions.
from bolmaqua import (
    CustomBoltzmannMachine,
    BM_SimAnn_Sampler,
    graph_to_bm,
    train_boltzmann_machine_pcd,
    sample_from_bm,
    GRID_SHAPE
)

def load_data():
    """
    Loads the 12x12 MNIST data (features and labels).
    Filters to keep only digits 0 and 1 for this test to match custom_BMs.py logic.
    """
    try:
        # Load training features and labels
        train_feats = np.load('mnist12x12_trainfeats.npy')
        train_labels = np.load('mnist12x12_trainlabels.npy')
    except FileNotFoundError:
        print("Error: Could not find mnist12x12 data files (trainfeats/trainlabels).")
        return None

    # Filter for digits 0 and 1 only (simplifying the problem for initial test)
    # This is consistent with the logic in custom_BMs.py which often filters for 0/1.
    mask = (train_labels == 0) | (train_labels == 1)
    train_feats = train_feats[mask]
    
    # Binarize features (threshold at 0.5)
    train_feats = (train_feats > 0.5).astype(np.float32)
    
    return torch.from_numpy(train_feats)

def create_rbm_graph(num_visible, num_hidden):
    """
    Creates a bipartite graph for an RBM.
    
    Nodes:
      - Visible: 0 to num_visible-1
      - Hidden: num_visible to num_visible+num_hidden-1
      
    Edges:
      - All pairs (v, h) where v is visible and h is hidden.
      - No intra-layer connections.
    """
    G = nx.Graph()
    
    visible_nodes = list(range(num_visible))
    hidden_nodes = list(range(num_visible, num_visible + num_hidden))
    
    G.add_nodes_from(visible_nodes)
    G.add_nodes_from(hidden_nodes)
    
    # Complete bipartite connections
    edges = [(v, h) for v in visible_nodes for h in hidden_nodes]
    G.add_edges_from(edges)
    
    # Node labels dictionary for the CustomBoltzmannMachine
    node_labels = {}
    for n in visible_nodes:
        node_labels[n] = 'visible'
    for n in hidden_nodes:
        node_labels[n] = 'hidden'
        
    return G, node_labels


print("==========================================")
print("Testing Custom BM Implementation with RBM Architecture (Bipartite Graph)")
print("==========================================")

#%% 1. Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1b. Hyperparameters (all in one place)
# Model architecture
num_visible = GRID_SHAPE[0] * GRID_SHAPE[1]  # Should be 144 for 12x12
num_hidden = 64

# Training
lr = 1e-4
weight_decay = 0.001
batch_size = 64
epochs = 30
k_steps = 10
persistent_chains = True

# End-of-training sampling: Gibbs
num_samples = 9
gibbs_burn_in = 1000

# End-of-training sampling: Simulated Annealing
sa_start_temp = 10.0
sa_end_temp = 0.2
sa_iterations = 8
sa_track_best = True

#%% 2. Data Loading
print("Loading 12x12 MNIST data (0s and 1s)...")
data = load_data()


dataset = TensorDataset(data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(f"Data samples: {len(data)}")
print(f"Visible units: {num_visible}")
print(f"Hidden units: {num_hidden}")

#%% 3. Graph Construction (RBM)
print("Constructing RBM (Complete Bipartite) Graph...")
G, node_labels = create_rbm_graph(num_visible, num_hidden)

#%% 4. Model Initialization
print("Initializing CustomBoltzmannMachine...")
model = graph_to_bm(G, node_labels)
model.to(device)

#%% 5. Training (PCD)
print(f"Starting PCD Training (lr={lr}, epochs={epochs}, k={k_steps})...")
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# train_boltzmann_machine_pcd handles the training loop
# Note: step_size overrides optimizer's LR inside the function
training_history = train_boltzmann_machine_pcd(
    model, 
    loader, 
    optimizer, 
    num_epochs=epochs, 
    k_steps=k_steps, 
    batch_size=batch_size, 
    step_size=lr,
    persistent=persistent_chains,
    train_data=data,
    eval_every=5
)

#%% 5b. Plot training metrics
if training_history is not None and 'pcd_loss' in training_history:
    epochs_range = range(1, len(training_history['pcd_loss']) + 1)

    # Determine how many reconstruction eval points we have
    has_train_recon = len(training_history.get('train_recon_mse', [])) > 0
    num_plots = 2 if has_train_recon else 1

    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    # --- Panel 1: PCD Loss & PLL ---
    ax1 = axes[0]
    color_loss = 'tab:blue'
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PCD Loss", color=color_loss)
    ax1.plot(epochs_range, training_history['pcd_loss'],
             marker='o', linewidth=2, color=color_loss, label='PCD Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    if 'pll' in training_history and len(training_history['pll']) > 0:
        ax1b = ax1.twinx()
        color_pll = 'tab:red'
        ax1b.set_ylabel("Pseudo Log-Likelihood", color=color_pll)
        ax1b.plot(epochs_range, training_history['pll'],
                  marker='s', linewidth=2, color=color_pll, label='PLL')
        ax1b.tick_params(axis='y', labelcolor=color_pll)
    ax1.set_title("PCD Loss & PLL")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Reconstruction metrics (if available) ---
    if has_train_recon:
        ax2 = axes[1]
        # Reconstruction metrics are computed every eval_every epochs + last epoch
        eval_every = 5
        n_recon = len(training_history['train_recon_mse'])
        # Build the epoch indices where recon was evaluated
        recon_epochs = [e for e in range(1, len(training_history['pcd_loss']) + 1)
                        if e % eval_every == 0 or e == len(training_history['pcd_loss'])]
        recon_epochs = recon_epochs[:n_recon]  # safety trim

        ax2.plot(recon_epochs, training_history['train_recon_mse'],
                 marker='o', linewidth=2, color='tab:green', label='MSE')
        ax2.plot(recon_epochs, training_history['train_recon_bce'],
                 marker='^', linewidth=2, color='tab:orange', label='BCE')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Reconstruction Loss")
        ax2.set_title("Train Reconstruction Metrics")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        # Accuracy on right y-axis
        ax2b = ax2.twinx()
        ax2b.plot(recon_epochs, training_history['train_recon_acc'],
                  marker='D', linewidth=2, color='tab:purple', label='Accuracy')
        ax2b.set_ylabel("Accuracy", color='tab:purple')
        ax2b.tick_params(axis='y', labelcolor='tab:purple')
        ax2b.legend(loc='upper right')

    fig.suptitle("Training Metrics")
    fig.tight_layout()
    fig.savefig("rbm_training_loss.png")
    print("Saved training metrics plot to rbm_training_loss.png")
    plt.show()

#%% 6. Sampling & Visualization
print("Generating samples from trained model...")
samples = sample_from_bm(model, num_samples=num_samples, burn_in_steps=gibbs_burn_in, method='gibbs')

# Plotting results
samples_np = samples.cpu().detach().numpy()

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle(f"RBM Samples (Epochs={epochs}, Hidden={num_hidden})")

for i, ax in enumerate(axes.flat):
    if i < len(samples_np):
        img = samples_np[i].reshape(GRID_SHAPE)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    
plt.tight_layout()
plt.savefig("rbm_test_samples.png")
print("Saved samples to rbm_test_samples.png")
plt.show()

#%% 7. Simulated Annealing Sampling & Visualization
print("Generating samples with BM_SimAnn_Sampler...")
sa_samples = BM_SimAnn_Sampler(
    model=model,
    start_temp=sa_start_temp,
    end_temp=sa_end_temp,
    max_iterations=sa_iterations,
    num_samples=num_samples,
    track_best=sa_track_best,
    verbose=True,
)

sa_samples_np = sa_samples.cpu().detach().numpy()

fig_sa, axes_sa = plt.subplots(4, 4, figsize=(8, 8))
fig_sa.suptitle(
    f"RBM Simulated Annealing Samples (Epochs={epochs}, Hidden={num_hidden})"
)

for i, ax in enumerate(axes_sa.flat):
    if i < len(sa_samples_np):
        img = sa_samples_np[i].reshape(GRID_SHAPE)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')

plt.tight_layout()
plt.savefig("rbm_test_samples_simann.png")
print("Saved SA samples to rbm_test_samples_simann.png")
plt.show()

print("Test complete. Window closed.")



# %%
