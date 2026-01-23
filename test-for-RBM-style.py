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



import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Import from bolmaqua
# We assume bolmaqua.py is in the same directory and contains the necessary classes and functions.
from bolmaqua import (
    CustomBoltzmannMachine,
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

def main():
    print("==========================================")
    print("Testing Custom BM Implementation with RBM Architecture (Bipartite Graph)")
    print("==========================================")
    
    # 1. Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Standard MNIST 12x12 size
    num_visible = GRID_SHAPE[0] * GRID_SHAPE[1]  # Should be 144
    num_hidden = 64  # Number of hidden units for the RBM
    
    # Training Hyperparameters
    lr = 0.005
    batch_size = 64
    epochs = 10
    k_steps = 5  # CD-k / PCD-k steps
    
    # 2. Data Loading
    print("Loading 12x12 MNIST data (0s and 1s)...")
    data = load_data()
    if data is None:
        return
    
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"Data samples: {len(data)}")
    print(f"Visible units: {num_visible}")
    print(f"Hidden units: {num_hidden}")
    
    # 3. Graph Construction (RBM)
    print("Constructing RBM (Complete Bipartite) Graph...")
    G, node_labels = create_rbm_graph(num_visible, num_hidden)
    
    # 4. Model Initialization
    print("Initializing CustomBoltzmannMachine...")
    model = graph_to_bm(G, node_labels)
    model.to(device)
    
    # 5. Training (PCD)
    print(f"Starting PCD Training (lr={lr}, epochs={epochs}, k={k_steps})...")
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # train_boltzmann_machine_pcd handles the training loop
    # Note: step_size overrides optimizer's LR inside the function
    train_boltzmann_machine_pcd(
        model, 
        loader, 
        optimizer, 
        num_epochs=epochs, 
        k_steps=k_steps, 
        batch_size=batch_size, 
        step_size=lr
    )
    
    # 6. Sampling & Visualization
    print("Generating samples from trained model...")
    num_samples = 16
    burn_in = 2000
    samples = sample_from_bm(model, num_samples=num_samples, burn_in_steps=burn_in, method='gibbs')
    
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
    print("Test complete. Window closed.")

if __name__ == "__main__":
    main()
