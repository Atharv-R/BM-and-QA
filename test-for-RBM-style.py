import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import json
import itertools
from torch.utils.data import TensorDataset, DataLoader

# Import from bolmaqua
from bolmaqua import (
    CustomBoltzmannMachine,
    graph_to_bm,
    train_boltzmann_machine_pcd,
    sample_from_bm,
    GRID_SHAPE,
    compute_pseudolikelihood # directly import if available
)

# -----------------------------------------------------------------------------
# 1. Config & Data Loading
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "hyperparam_opt_results_jan22.json"
MAX_RUNTIME_SEC = 1*60*60  # max rutime in seconds
OUTPUT_IMG = "best_model_samples.png"

def load_data_custom_style():
    """
    Loads data matching the logic in custom_BMs.py:
    - Filter 0/1
    - Subset first 200
    - Invert bits (0->1, 1->0) using < 0.5 logic
    """
    print("Loading dataset...")
    try:
        train_feats = np.load('mnist12x12_trainfeats.npy')
        train_labels = np.load('mnist12x12_trainlabels.npy')
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Filter 0/1
    mask = (train_labels == 0) | (train_labels == 1)
    train_feats = train_feats[mask]
    
    # # Subset to 200 (Train + Val)
    # train_feats = train_feats[:200]
    
    # Normalize
    train_feats = (train_feats - train_feats.min()) / (train_feats.max() - train_feats.min())
    
    # Binarize & Invert (custom_BMs logic: < 0.5 becomes 1)
    # Important: This matches the 'other setting' logic exactly
    data = (torch.from_numpy(train_feats).float() < 0.5).float()
    
    # Split Train/Val (80/20)
    # 200 samples -> 160 train, 40 val
    n_total = len(data)
    n_train = int(0.8 * n_total)
    
    # Shuffle before split (ensure randomness)
    perm = torch.randperm(n_total)
    data = data[perm]
    
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    print(f"Data Loaded: {len(train_data)} Train, {len(val_data)} Val")
    return train_data, val_data

def create_rbm_graph(num_visible, num_hidden):
    G = nx.Graph()
    visible_nodes = list(range(num_visible))
    hidden_nodes = list(range(num_visible, num_visible + num_hidden))
    G.add_nodes_from(visible_nodes)
    G.add_nodes_from(hidden_nodes)
    G.add_edges_from([(v, h) for v in visible_nodes for h in hidden_nodes])
    
    node_labels = {n: 'visible' for n in visible_nodes}
    node_labels.update({n: 'hidden' for n in hidden_nodes})
    return G, node_labels

# -----------------------------------------------------------------------------
# 2. Evaluation Metric
# -----------------------------------------------------------------------------

def evaluate_pseudolikelihood(model, val_loader):
    """
    Computes average Pseudolikelihood (computed via bolmaqua helper) on validation set.
    """
    model.eval()
    total_pll = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # batch is a list [data] because TensorDataset returns tuple
            v_input = batch[0].to(device)
            # Use bolmaqua helper
            pll = compute_pseudolikelihood(model, v_input)
            total_pll += pll
            count += 1
            
    return total_pll / count

# -----------------------------------------------------------------------------
# 3. Search Space
# -----------------------------------------------------------------------------

def get_random_params():
    # Random Search Logic
    
    # 1. Learning Rate: LogUniform between 0.001 and 0.05
    lr = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.05))))
    
    # 2. L2 Reg: LogUniform between 1e-5 and 0.01
    l2 = float(np.exp(np.random.uniform(np.log(0.00001), np.log(0.01))))
    
    # 3. Batch Size: Discrete
    batch_size = int(random.choice([16, 32, 64]))
    
    # 4. Steps k: Discrete
    # Bias slightly towards lower k for speed, check if high k helps mixing
    k_steps = int(random.choice([1, 1, 5, 10]))
    
    # 5. Epochs: Discrete (Varying as requested)
    epochs = int(random.choice([10, 20, 40, 60, 80]))
    
    return {
        "lr": round(lr, 5),
        "l2": round(l2, 5),
        "batch_size": batch_size,
        "k_steps": k_steps,
        "epochs": epochs
    }

# -----------------------------------------------------------------------------
# 4. Main Search Loop
# -----------------------------------------------------------------------------

def main():
    print(f"Starting Hyperparameter Search (Max {MAX_RUNTIME_SEC}s)...")
    
    # Load Data
    train_data, val_data = load_data_custom_style()
    if train_data is None: return

    # Validation loader (Batch size usually doesn't affect metric value, just speed)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=20, shuffle=False)
    
    # Fixed RBM Architecture
    num_visible = GRID_SHAPE[0] * GRID_SHAPE[1]
    num_hidden = 50 # Fixed to match custom_BMs
    
    G, node_labels = create_rbm_graph(num_visible, num_hidden)
    
    start_time = time.time()
    results = []
    
    run_id = 0
    best_score = -float('inf') 
    best_params = None

    while (time.time() - start_time) < MAX_RUNTIME_SEC:
        run_id += 1
        params = get_random_params()
        
        print(f"\n--- Run {run_id} ---")
        print(f"Params: {json.dumps(params, indent=2)}")
        
        # 1. Init Model
        model = graph_to_bm(G, node_labels)
        
        # 2. Optim
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'], weight_decay=params['l2'])
        
        # 3. Train
        # Only use valid batch sizes (<= num samples)
        bs = min(params['batch_size'], len(train_data))
        train_loader = DataLoader(TensorDataset(train_data), batch_size=bs, shuffle=True, drop_last=False)
        
        try:
             # This will print per-epoch logs from bolmaqua.py
             train_boltzmann_machine_pcd(
                model, 
                train_loader, 
                optimizer, 
                num_epochs=params['epochs'], 
                k_steps=params['k_steps'], 
                batch_size=bs, 
                step_size=params['lr']
            )
        except Exception as e:
            print(f"Training Failed: {e}")
            continue

        # 4. Evaluate
        try:
            score = evaluate_pseudolikelihood(model, val_loader)
        except Exception as e:
             print(f"Eval warning: {e}")
             score = -9999.0

        print(f"Result (Avg PLL): {score:.4f}")
        
        result_entry = {
            "run_id": run_id,
            "params": params,
            "score": float(score),
            "metric": "PseudoLikelihood (Higher Better)"
        }
        results.append(result_entry)
        
        # 5. Check Best
        if score > best_score:
            best_score = score
            best_params = params
            
            # Save samples from best model immediately
            print(f"New Best! Saving samples to {OUTPUT_IMG}")
            try:
                samples = sample_from_bm(model, num_samples=16, burn_in_steps=1000, method='gibbs')
                samples_np = samples.cpu().detach().numpy()
                
                fig, axes = plt.subplots(4, 4, figsize=(6, 6))
                for i, ax in enumerate(axes.flat):
                    if i < len(samples_np):
                        ax.imshow(samples_np[i].reshape(GRID_SHAPE), cmap='gray', vmin=0, vmax=1)
                    ax.axis('off')
                plt.suptitle(f"Best Run {run_id}: PLL={score:.2f}")
                plt.tight_layout()
                plt.savefig(OUTPUT_IMG)
                plt.close()
            except Exception as e:
                print(f"Plotting failed: {e}")

    # End Search
    print("\n==========================================")
    print("Search Complete.")
    print("==========================================")
    print(f"Best Score: {best_score}")
    print(f"Best Params: {json.dumps(best_params, indent=2)}")
    
    # Save all results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()


