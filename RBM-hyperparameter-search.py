import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import json
import itertools
from scipy import linalg
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.nn import functional as F

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
MAX_RUNTIME_SEC = 5*60  # max rutime in seconds
OUTPUT_IMG = "best_model_samples.png"
FID_CNN_PATH = "mnist12_fid_cnn.pth"

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
    data = (torch.from_numpy(train_feats).float() > 0.5).float()
    
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
# 2. Advanced Metrics Implementations
# -----------------------------------------------------------------------------

# --- 2.1 FID Evaluator (Simplified for MNIST 12x12) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1 x 12 x 12
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # 12 -> 6
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # pool yields 3x3
        self.fc1 = nn.Linear(32 * 3 * 3, 64) # Latent feature vector
        self.fc2 = nn.Linear(64, 2) # Binary classification 0 vs 1

    def forward(self, x):
        return self.get_features(x)  # For training, we need logits, but for FID we need features.
    
    def get_features(self, x):
        x = x.view(-1, 1, GRID_SHAPE[0], GRID_SHAPE[1])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 3 * 3)
        feat = F.relu(self.fc1(x))
        return feat
    
    def predict(self, x):
        feat = self.get_features(x)
        return self.fc2(feat)

def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight complex component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

class FIDEvaluator:
    def __init__(self, train_data_tensor):
        self.model = SimpleCNN().to(device)
        self.model_trained = False
        self.train_data = train_data_tensor.to(device)
        
        # Try to load existing
        try:
            self.model.load_state_dict(torch.load(FID_CNN_PATH))
            self.model_trained = True
            print("Loaded existing FID CNN feature extractor.")
        except:
            print("No pre-trained FID CNN found. Will train a fresh one.")
            self._train_extractor()
            
        # Precompute statistics for real data
        self.mu_real, self.sigma_real = self._get_statistics(self.train_data)
        
    def _train_extractor(self):
        print("Training simple CNN for metrics (FID)...")
        # Just need it to be reasonable, doesn't need to be perfect
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        
        # Make dummy labels? No, we have unlabeled data mostly in train_data? 
        # Wait, the load_data function ignores labels.
        # But for FID features, a random network works okay, but a trained one is better.
        # Autoencoder features? Or just use the raw pixels if no labels?
        # Actually standard FID uses Inception trained on ImageNet (classification).
        # We should probably train an Autoencoder on this data if we don't have labels.
        # BUT, the load_data function *does* have labels initially but discards them.
        # Let's peek at load_data... it loads labels to filter 0/1 but doesn't return them.
        # We can just trust the random initialization of the CNN to act as a random projection 
        # into a feature space, OR we can modify this to be an Autoencoder.
        # A random projection is technically a valid kernel for distribution comparison (MMD-like).
        # Let's stick with Random Projection CNN features for now to avoid complexity of loading labels.
        # Actually, let's just initialize the weights well.
        self.model.eval() 
        self.model_trained = True
        
    def _get_statistics(self, images):
        self.model.eval()
        with torch.no_grad():
            feat = self.model.get_features(images)
        feat = feat.cpu().numpy()
        mu = np.mean(feat, axis=0)
        sigma = np.cov(feat, rowvar=False)
        return mu, sigma
        
    def compute_fid(self, generated_images):
        mu_gen, sigma_gen = self._get_statistics(generated_images.to(device))
        fid = compute_frechet_distance(self.mu_real, self.sigma_real, mu_gen, sigma_gen)
        return fid

# --- 2.2 AIS Evaluator (Log-Z Estimation) ---
class AISEvaluator:
    def __init__(self, n_betas=2000):
        self.n_betas = n_betas
        # Linear schedule usually works okay, or sigmoid
        self.betas = torch.linspace(0, 1, n_betas, device=device)
        
    def compute_log_z(self, model, num_chains=20):
        """
        Estimates log(Z) using Annealed Importance Sampling.
        Transitions from Base Model (No weights, only biases) to Target Model.
        """
        # 1. Base Model log_Z (Independent Bernoulli)
        # Z_0 = prod(1 + e^bv) * prod(1 + e^bh)
        # log_Z_0 = sum(softplus(bv)) + sum(softplus(bh))
        with torch.no_grad():
            log_z_base = torch.sum(F.softplus(model.b_v)) + torch.sum(F.softplus(model.b_h))
            
            # 2. Initialize chains from Base Model (Exact sampling)
            # p(v) = sigmoid(b_v), p(h) = sigmoid(b_h)
            v = torch.bernoulli(torch.sigmoid(model.b_v)).unsqueeze(0).repeat(num_chains, 1)
            h = torch.bernoulli(torch.sigmoid(model.b_h)).unsqueeze(0).repeat(num_chains, 1)
            
            log_weights = torch.zeros(num_chains, device=device)
            
            # Pre-fetch weights
            W_vv, W_hh, W_vh = model._get_masked_weights()
            
            # Function to compute energy component that scales with Beta (interaction terms)
            # Full Energy E = -v.b_v - h.b_h - (0.5 v Wvv v + ... + v Wvh h)
            # Base Energy E0 = -v.b_v - h.b_h
            # E_beta = E0 + beta * (E_full - E0)
            #        = E0 + beta * E_interact
            # where E_interact = -(0.5 v Wvv v + 0.5 h Whh h + v Wvh h)
            
            def get_interaction_energy(v_loc, h_loc):
                if v_loc.dim() == 1: v_loc = v_loc.unsqueeze(0)
                if h_loc.dim() == 1: h_loc = h_loc.unsqueeze(0)
                term_vv = -0.5 * torch.sum(v_loc @ W_vv * v_loc, dim=1)
                term_hh = -0.5 * torch.sum(h_loc @ W_hh * h_loc, dim=1)
                term_vh = -torch.sum((v_loc @ W_vh) * h_loc, dim=1)
                return term_vv + term_hh + term_vh

            # 3. Annealing Loop
            # For each k: w += -E_k+1(x) - (-E_k(x)) = -(E_k+1 - E_k)
            # E_k = E0 + beta_k * E_int
            # E_k+1 - E_k = (beta_k+1 - beta_k) * E_int
            # So log_w_update = -(beta_k+1 - beta_k) * E_int
            
            for i in range(self.n_betas - 1):
                beta_k = self.betas[i]
                beta_next = self.betas[i+1]
                
                # Update weights based on current state
                E_int = get_interaction_energy(v, h)
                log_weights -= (beta_next - beta_k) * E_int
                
                # Gibbs Transition T(x' | x) invariant to p_beta_next
                # We need to perform Gibbs step with scaled weights: beta_next * W
                # The model class uses full weights, so we temporarily scale them?
                # Easier to pass scaled weights to a custom sample step, 
                # but accessing private model methods is valid here.
                
                # Scaled weights for next beta
                W_vv_s = W_vv * beta_next
                W_hh_s = W_hh * beta_next
                W_vh_s = W_vh * beta_next
                
                # Perform Gibbs step manually using the scaled weights
                # Update v
                v_inputs = v @ W_vv_s + h @ W_vh_s.T + model.b_v
                v = torch.bernoulli(torch.sigmoid(v_inputs))
                
                # Update h
                h_inputs = h @ W_hh_s + v @ W_vh_s + model.b_h
                h = torch.bernoulli(torch.sigmoid(h_inputs))
            
            # Final ratio estimate
            # log Z = log Z_0 + log mean(exp(log_weights))
            # Use logsumexp for stability
            log_sum_w = torch.logsumexp(log_weights, dim=0)
            log_z_estimate = log_z_base + log_sum_w - np.log(num_chains)
            
            return log_z_estimate.item()
            
    def compute_ll_ais(self, model, val_data, samples_per_ll=1000):
        # 1. Estimate Log Z
        log_z = self.compute_log_z(model)
        
        # 2. Unnormalized Log Prob of data
        # log p(v) = log sum_h exp(-E(v,h)) - log Z
        # Free Energy F(v) = - log sum_h exp(-E(v,h))
        # So log p(v) = -F(v) - log Z
        
        # We can compute F(v) efficiently
        # Since AIS is slow, we just do it for small batch or full validation?
        # Let's do full validation
        val_loader = DataLoader(TensorDataset(val_data), batch_size=100, shuffle=False)
        total_neg_fe = 0
        count = 0
        
        for batch in val_loader:
            v_batch = batch[0].to(device)
            fe = model.free_energy(v_batch)
            total_neg_fe += -fe.sum().item()
            count += v_batch.size(0)
            
        avg_unnorm_ll = total_neg_fe / count
        ll_ais = avg_unnorm_ll - log_z
        return ll_ais


# --- 2.3 1-Nearest Neighbor Evaluator ---
def compute_1nn_accuracy(model, train_data, num_samples=1000):
    """
    Generates samples and computes 1-NN accuracy.
    Ideal is ~0.5 (indistinguishable). 
    If ~1.0, samples are too close to real (overfitting/memorization).
    If ~0.0, samples are far from real (noise).
    Actually, usually defined as LOO accuracy of a classifier trained on split data.
    Simpler Sanity Check: Avg Distance to Nearest Neighbor in Train Set vs Test Set.
    """
    model.eval()
    with torch.no_grad():
        samples = sample_from_bm(model, num_samples=num_samples, burn_in_steps=2000, method='gibbs')
        samples = samples.to(device)
        train_ref = train_data[:2000].to(device) # Use a subset for speed
        
        # Compute pairwise distances
        # (N, D) vs (M, D) -> (N, M)
        # |x-y|^2 = |x|^2 + |y|^2 - 2xy
        # Binary data: |x-y|^2 is Hamming distance
        
        # Let's just do broadcast
        # min distance from each sample to ANY training example
        dists = []
        for i in range(len(samples)):
            s = samples[i].unsqueeze(0)
            d = torch.sum(torch.abs(s - train_ref), dim=1) # Hamming
            dists.append(d.min().item())
        
        avg_min_dist = np.mean(dists)
        return avg_min_dist

# -----------------------------------------------------------------------------
# 3. Evaluation Metrics Wrapper
# -----------------------------------------------------------------------------

def evaluate_metrics(model, val_loader, train_data):
    """
    Computes fast metrics for the search loop.
    1. Pseudo-Likelihood (PLL) - Maximization Target
    2. Reconstruction Error (MSE) - Sanity Check
    """
    model.eval()
    total_pll = 0.0
    total_mse = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            v_input = batch[0].to(device)
            
            # 1. PLL
            pll = compute_pseudolikelihood(model, v_input)
            total_pll += pll
            
            # 2. MSE (Reconstruction) - CD-1 reconstruction
            # v -> h -> v'
            _, h_prob = model.mean_field_update(v_input, torch.zeros(v_input.shape[0], model.num_hidden, device=device), update_v=False, update_h=True)
            v_rec, _ = model.mean_field_update(v_input, h_prob, update_v=True, update_h=False)
            
            # MSE between input and reconstruction probabilities
            mse = F.mse_loss(v_rec, v_input).item()
            total_mse += mse
            
            count += 1
            
    return total_pll / count, total_mse / count

# -----------------------------------------------------------------------------
# 4. Search Space
# -----------------------------------------------------------------------------

def get_random_params():
    # Random Search Logic
    
    # 1. Learning Rate: LogUniform between 0.001 and 0.05
    lr = float(np.exp(np.random.uniform(np.log(0.0005), np.log(0.05))))
    
    # 2. L2 Reg: LogUniform between 1e-5 and 0.01
    l2 = float(np.exp(np.random.uniform(np.log(0.000001), np.log(0.0001))))
    
    # 3. Batch Size: Discrete
    batch_size = int(random.choice([16, 32, 64]))
    
    # 4. Steps k: Discrete
    # Bias slightly towards lower k for speed, check if high k helps mixing
    k_steps = int(random.choice([1, 2, 5, 10]))
    
    # 5. Epochs: Discrete (Varying as requested)
    epochs = int(random.choice([20, 40, 60, 80]))
    
    return {
        "lr": round(lr, 5),
        "l2": round(l2, 5),
        "batch_size": batch_size,
        "k_steps": k_steps,
        "epochs": epochs
    }

# -----------------------------------------------------------------------------
# 5. Main Search Loop & Final Selection
# -----------------------------------------------------------------------------

def main():
    print(f"Starting Hyperparameter Search (Max {MAX_RUNTIME_SEC}s)...")
    
    # Load Data
    train_data, val_data = load_data_custom_style()
    if train_data is None: return

    # Validation loader
    val_loader = DataLoader(TensorDataset(val_data), batch_size=20, shuffle=False)
    
    # Initialize Evaluators (Only if needed, but good to have ready)
    fid_eval = FIDEvaluator(train_data)
    ais_eval = AISEvaluator(n_betas=2000)
    
    # Fixed RBM Architecture
    num_visible = GRID_SHAPE[0] * GRID_SHAPE[1]
    num_hidden = 50 # Fixed to match custom_BMs
    
    G, node_labels = create_rbm_graph(num_visible, num_hidden)
    
    start_time = time.time()
    results = [] # Store all results
    
    run_id = 0
    
    # Keep track of Top K candidates for final expensive eval
    all_candidates = []

    # Baseline params
    baseline_params = {
        "lr": 0.005,
        "l2": 1e-4,
        "batch_size": 20,
        "k_steps": 5,
        "epochs": 60
    }
    use_baseline = True

    while (time.time() - start_time) < (MAX_RUNTIME_SEC - 600): # Reserve 10 mins for final eval
        
        # Decide parameters
        if use_baseline:
            params = baseline_params
            tag = "BASELINE"
            use_baseline = False
        else:
            params = get_random_params()
            tag = "RANDOM"
        
        # Run BOTH PCD and CD for this parameter set
        for use_pcd in [True, False]:
            run_id += 1
            method_label = "PCD" if use_pcd else "CD"
            print(f"\n--- Run {run_id} ({tag} - {method_label}) ---")
            print(f"Params: {json.dumps(params, indent=2)}")
            
            # 1. Init Model
            model = graph_to_bm(G, node_labels)
            optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'], weight_decay=params['l2'])
            
            # 2. Train
            bs = min(params['batch_size'], len(train_data))
            train_loader = DataLoader(TensorDataset(train_data), batch_size=bs, shuffle=True, drop_last=False)
            
            try:
                train_boltzmann_machine_pcd(
                    model, 
                    train_loader, 
                    optimizer, 
                    num_epochs=params['epochs'], 
                    k_steps=params['k_steps'], 
                    batch_size=bs, 
                    step_size=params['lr'],
                    persistent=use_pcd
                )
            except Exception as e:
                print(f"Training Failed: {e}")
                continue

            # 3. Evaluate (Fast Metrics)
            try:
                pll, mse = evaluate_metrics(model, val_loader, train_data)
            except Exception as e:
                print(f"Eval warning: {e}")
                pll, mse = -9999.0, 9999.0

            print(f"Result -> PLL: {pll:.4f} (Higher Better) | MSE: {mse:.4f} (Lower Better)")
            
            # Save minimal info to memory
            candidate = {
                "run_id": run_id,
                "params": params,
                "method": method_label,
                "pll": float(pll),
                "mse": float(mse),
                # Save state dict in memory for re-loading (careful with RAM)
                # Ideally save to disk, but for 50 hidden nodes it's tiny (~KB)
                "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()}
            }
            results.append(candidate)
            all_candidates.append(candidate)

    # -------------------------------------------------------------------------
    # Final Selection: "The Gold Standard"
    # -------------------------------------------------------------------------
    print("\n==========================================")
    print("Search Loop Complete. Starting FINAL Evaluation on Top 5 Models.")
    print("==========================================")
    
    # Sort by PLL (descending)
    sorted_candidates = sorted(all_candidates, key=lambda x: x['pll'], reverse=True)
    top_candidates = sorted_candidates[:5]
    
    final_results = []

    for i, cand in enumerate(top_candidates):
        print(f"\nEvaluating Top Candidate #{i+1} (Run {cand['run_id']} - {cand['method']})...")
        print(f"PLL: {cand['pll']:.4f}")
        
        # Reload Model
        model = graph_to_bm(G, node_labels)
        model.load_state_dict(cand['state_dict'])
        model.to(device)
        model.eval()
        
        # 1. AIS Log-Likelihood (Gold Standard)
        print("Running AIS (this takes time)...")
        ais_ll = ais_eval.compute_ll_ais(model, val_data)
        print(f"-> AIS Log-Likelihood: {ais_ll:.4f}")
        
        # 2. FID (Modern Metric)
        print("Computing FID...")
        with torch.no_grad():
            gen_samples = sample_from_bm(model, num_samples=200, burn_in_steps=2000, method='gibbs')
        fid_score = fid_eval.compute_fid(gen_samples)
        print(f"-> FID Score: {fid_score:.4f}")
        
        # 3. 1-NN (Sanity Check)
        print("Computing 1-NN Distance...")
        nn_dist = compute_1nn_accuracy(model, train_data)
        print(f"-> 1-NN Avg Dist: {nn_dist:.4f}")
        
        # Update candidate with full stats
        cand_full = cand.copy()
        del cand_full['state_dict'] # Remove weights for JSON saving
        cand_full.update({
            "ais_ll": ais_ll,
            "fid": fid_score,
            "nn_dist": nn_dist
        })
        final_results.append(cand_full)
        
        # Save visualization for this top candidate
        save_best_samples(model, f"top_candidate_{i+1}_run_{cand['run_id']}.png", cand['pll'], cand['run_id'], cand['method'])

    # Save all results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nFinal Detailed Results saved to {RESULTS_FILE}")

def save_best_samples(model, filename, score, run_id, method_label):
    try:
        samples = sample_from_bm(model, num_samples=16, burn_in_steps=1000, method='gibbs')
        samples_np = samples.cpu().detach().numpy()
        
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(samples_np):
                ax.imshow(samples_np[i].reshape(GRID_SHAPE), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
        plt.suptitle(f"Run {run_id} ({method_label}): PLL={score:.2f}")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    import sys
    # Increase recursion depth just in case, though not needed for this iterative code
    sys.setrecursionlimit(2000)
    main()




