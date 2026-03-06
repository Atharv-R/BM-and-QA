"""
eval_quality.py — Sample quality evaluation for Boltzmann Machines on MNIST

Provides:
  1. A small CNN classifier trained on binarized 28x28 MNIST (all 10 digits).
     Trained once and cached to disk.
  2. FID (Fréchet Inception Distance) computed in the classifier's feature space,
     with bootstrap confidence intervals.
  3. Additional per-sample quality metrics: classification confidence, class balance.
  4. eval_samples_fullMNIST(): the main entry point — generates samples via
     multiple sampling methods, evaluates each, and returns a structured report.

Usage:
    from eval_quality import eval_samples_fullMNIST
    report = eval_samples_fullMNIST(model, real_data, grid_shape=(28,28),
                                     num_samples=300)

The classifier is deliberately simple (3-layer CNN, ~15K params) because:
  - We're evaluating binarized 28x28 images, not natural photos.
  - It needs to be fast to train (< 1 min on GPU) and cheap to run.
  - FID in this feature space captures digit structure well enough for ranking
    hyperparameter configurations.
"""

import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from scipy import linalg

# ---------------------------------------------------------------------------
# Device (matches bolmaqua convention)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default path for the cached classifier
_DEFAULT_CLASSIFIER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "mnist_quality_classifier.pt"
)


# =====================================================================
# 1.  MNIST Classifier
# =====================================================================

class MNISTQualityClassifier(nn.Module):
    """
    Small CNN for binarized 28x28 MNIST (10 classes).

    Architecture:
        conv1 (1→16, 3x3, stride 2) → ReLU → conv2 (16→32, 3x3, stride 2) →
        ReLU → conv3 (32→64, 3x3, stride 2) → ReLU → AdaptiveAvgPool(2x2) →
        flatten (feature_dim=256) → fc (256→10)

    The 256-d feature vector (after the pool, before the fc) is used as the
    "inception-like" embedding for FID computation.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)   # 28→14
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 14→7
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 7→4
        self.pool = nn.AdaptiveAvgPool2d(2)                      # 4→2
        self.fc = nn.Linear(64 * 2 * 2, 10)
        self.feature_dim = 256  # 64 * 2 * 2

    def features(self, x):
        """Return the 256-d feature vector (before the classification head)."""
        if x.dim() == 2:
            # Flat (B, 784) → (B, 1, 28, 28)
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # (B, 256)
        return x

    def forward(self, x):
        feat = self.features(x)
        return self.fc(feat)

    def predict_proba(self, x):
        """Return softmax probabilities (B, 10)."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


def train_mnist_classifier(
    save_path: str = _DEFAULT_CLASSIFIER_PATH,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
    binarize_threshold: float = 0.5,
    force_retrain: bool = False,
) -> MNISTQualityClassifier:
    """
    Train (or load from cache) the 10-class MNIST classifier on binarized data.

    If the cached checkpoint contains a 'config' key (from the classifier
    hyperparameter search), a FlexibleMNISTClassifier is loaded instead of
    the default MNISTQualityClassifier. Both expose the same .features() and
    .forward() interface.

    Returns the trained classifier on `device`, in eval mode.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check for cached model
    if os.path.exists(save_path) and not force_retrain:
        print(f"[eval_quality] Loading cached classifier from {save_path}")
        state = torch.load(save_path, map_location=device, weights_only=False)
        if 'config' in state:
            # Flexible classifier from hyperparameter search
            from classifier_hyperparam_search import FlexibleMNISTClassifier
            clf = FlexibleMNISTClassifier(state['config']).to(device)
            clf.load_state_dict(state['model_state_dict'])
            print(f"  Loaded FlexibleMNISTClassifier (feature_dim={clf.feature_dim}, "
                  f"acc={state.get('test_accuracy', '?')})")
        else:
            # Original fixed classifier
            clf = MNISTQualityClassifier().to(device)
            clf.load_state_dict(state['model_state_dict'])
            print(f"  Cached classifier accuracy: {state.get('test_accuracy', '?')}")
        clf.eval()
        return clf

    print("[eval_quality] Training MNIST quality classifier...")
    t0 = time.time()

    # Load MNIST
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    def binarize_dataset(ds):
        imgs = ds.data.float() / 255.0           # (N, 28, 28) in [0,1]
        imgs = (imgs >= binarize_threshold).float()  # binarize
        imgs = imgs.unsqueeze(1)                  # (N, 1, 28, 28)
        labels = ds.targets
        return imgs, labels

    train_imgs, train_labels = binarize_dataset(train_ds)
    test_imgs, test_labels = binarize_dataset(test_ds)

    train_loader = DataLoader(
        TensorDataset(train_imgs, train_labels),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_imgs, test_labels),
        batch_size=batch_size, shuffle=False,
    )

    clf = MNISTQualityClassifier().to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        clf.train()
        total_loss = 0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = clf(imgs)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        train_acc = correct / total
        if (epoch + 1) % 4 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/total:.4f}  train_acc={train_acc:.4f}")

    # Test accuracy
    clf.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = clf(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    test_acc = correct / total
    print(f"  Final test accuracy: {test_acc:.4f}  (training time: {time.time()-t0:.1f}s)")

    # Save
    torch.save({
        "model_state_dict": clf.state_dict(),
        "test_accuracy": test_acc,
        "binarize_threshold": binarize_threshold,
        "epochs": epochs,
    }, save_path)
    print(f"  Saved classifier to {save_path}")

    clf.eval()
    return clf


# =====================================================================
# 2.  FID Computation (with bootstrap confidence intervals)
# =====================================================================

def _compute_features(classifier: MNISTQualityClassifier, images: torch.Tensor,
                      batch_size: int = 512) -> np.ndarray:
    """
    Extract feature vectors from the classifier for a set of images.

    Args:
        classifier: Trained MNISTQualityClassifier (in eval mode).
        images: (N, 784) or (N, 28, 28) or (N, 1, 28, 28) tensor of binarized images.
        batch_size: Batch size for feature extraction.

    Returns:
        (N, 256) numpy array of feature vectors.
    """
    classifier.eval()
    all_feats = []
    images = images.to(device)
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            feats = classifier.features(batch)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def _pca_reduce(real_features: np.ndarray, gen_features: np.ndarray,
                n_components: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce feature dimensionality via PCA (fit on real data, apply to both).

    This stabilizes FID computation when the number of generated samples is
    small relative to the feature dimensionality (e.g., 300 samples in 256-d
    space). PCA is fit on real_features, ensuring the projection is stable.

    Args:
        real_features: (N_r, D) features from real data.
        gen_features: (N_g, D) features from generated data.
        n_components: Number of PCA components to keep.

    Returns:
        (real_reduced, gen_reduced): both of shape (N, n_components).
    """
    n_components = min(n_components, real_features.shape[1], real_features.shape[0])

    # Center on real data mean
    mu = np.mean(real_features, axis=0)
    real_centered = real_features - mu
    gen_centered = gen_features - mu

    # SVD on real data
    U, S, Vt = np.linalg.svd(real_centered, full_matrices=False)
    components = Vt[:n_components]  # (n_components, D)

    real_reduced = real_centered @ components.T
    gen_reduced = gen_centered @ components.T

    return real_reduced, gen_reduced


def compute_fid(real_features: np.ndarray, gen_features: np.ndarray,
                eps: float = 1e-6) -> float:
    """
    Compute the Fréchet Inception Distance between two sets of features.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r @ Sigma_g)^{1/2})

    Applies shrinkage regularization to covariance matrices to handle
    near-singular cases (common with binary image features).

    Args:
        real_features: (N_r, D) numpy array.
        gen_features: (N_g, D) numpy array.
        eps: Regularization strength for covariance diagonal.

    Returns:
        FID score (lower is better).
    """
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(gen_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(gen_features, rowvar=False)

    # Regularize covariance matrices (standard practice for FID)
    sigma_r += np.eye(sigma_r.shape[0]) * eps
    sigma_g += np.eye(sigma_g.shape[0]) * eps

    diff = mu_r - mu_g
    mean_term = diff @ diff

    # Matrix square root via scipy
    product = sigma_r @ sigma_g
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covmean_sq, info = linalg.sqrtm(product, disp=False)

    # Numerical stability: if imaginary component is small, discard it
    if np.iscomplexobj(covmean_sq):
        if np.max(np.abs(covmean_sq.imag)) < 1e-3 * np.max(np.abs(covmean_sq.real)):
            covmean_sq = covmean_sq.real
        else:
            # Stronger regularization fallback
            offset = np.eye(sigma_r.shape[0]) * eps * 10
            covmean_sq, _ = linalg.sqrtm((sigma_r + offset) @ (sigma_g + offset), disp=False)
            covmean_sq = covmean_sq.real

    trace_term = np.trace(sigma_r + sigma_g - 2.0 * covmean_sq)
    fid = float(mean_term + trace_term)

    return max(fid, 0.0)  # Clamp to non-negative (numerical noise)


def compute_fid_with_ci(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    n_bootstrap: int = 100,
    ci_level: float = 0.95,
    seed: int = 42,
    pca_components: int | None = 32,
) -> dict:
    """
    Compute FID with bootstrap confidence intervals.

    Optionally reduces features via PCA first (fit on real data) to stabilize
    covariance estimation when the number of generated samples is small.

    Bootstraps the generated features (resampling with replacement) to estimate
    the distribution of FID scores, then reports the point estimate and CI.

    Note on bias: FID has a known positive bias when N_gen is small relative
    to the feature dimension. With PCA-32 and N_gen=400, the real-vs-real
    baseline is ~100. This bias is constant across configurations, so relative
    rankings are valid. The noise baseline is ~7500, giving ~75x dynamic range.

    Args:
        real_features: (N_r, D) features from real data.
        gen_features: (N_g, D) features from generated samples.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (default 0.95 = 95% CI).
        seed: Random seed for reproducibility.
        pca_components: If set, reduce features to this many PCA dimensions
            before computing FID. Default 32, good for N_gen=400. Set to None
            to skip PCA.

    Returns:
        dict with keys: 'fid', 'ci_low', 'ci_high', 'ci_level', 'fid_std',
                         'pca_components'
    """
    rng = np.random.RandomState(seed)

    # Apply PCA if requested
    if pca_components is not None:
        real_reduced, gen_reduced = _pca_reduce(real_features, gen_features, pca_components)
    else:
        real_reduced, gen_reduced = real_features, gen_features

    n_gen = len(gen_reduced)

    # Point estimate using all generated features
    fid_point = compute_fid(real_reduced, gen_reduced)

    # Bootstrap: resample generated features
    bootstrap_fids = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_gen, size=n_gen, replace=True)
        gen_boot = gen_reduced[idx]
        fid_boot = compute_fid(real_reduced, gen_boot)
        bootstrap_fids.append(fid_boot)

    bootstrap_fids = np.array(bootstrap_fids)
    alpha = 1.0 - ci_level
    ci_low = float(np.percentile(bootstrap_fids, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_fids, 100 * (1 - alpha / 2)))

    return {
        "fid": fid_point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci_level,
        "fid_std": float(np.std(bootstrap_fids)),
        "pca_components": pca_components,
    }


# =====================================================================
# 3.  Classifier-based quality metrics
# =====================================================================

def compute_classifier_metrics(
    classifier: MNISTQualityClassifier,
    samples: torch.Tensor,
    confidence_threshold: float = 0.8,
) -> dict:
    """
    Compute sample quality metrics using the classifier.

    Args:
        classifier: Trained MNISTQualityClassifier.
        samples: (N, 784) tensor of generated binarized images.
        confidence_threshold: Threshold for "high confidence" classification.

    Returns:
        dict with:
          'mean_confidence': average max-class probability across samples
          'frac_high_confidence': fraction with max prob > threshold
          'predicted_classes': (N,) numpy array of predicted digit labels
          'class_distribution': dict mapping digit → fraction of samples
          'class_balance_entropy': entropy of class distribution (higher = more uniform)
          'per_class_confidence': dict mapping digit → mean confidence for that class
    """
    classifier.eval()
    samples = samples.to(device)

    with torch.no_grad():
        probs = classifier.predict_proba(samples)  # (N, 10)

    max_probs, pred_classes = probs.max(dim=1)
    max_probs_np = max_probs.cpu().numpy()
    pred_classes_np = pred_classes.cpu().numpy()
    probs_np = probs.cpu().numpy()

    # Overall confidence
    mean_confidence = float(np.mean(max_probs_np))
    frac_high_conf = float(np.mean(max_probs_np > confidence_threshold))

    # Class distribution
    class_counts = {}
    class_confidences = {}
    n = len(pred_classes_np)
    for d in range(10):
        mask = pred_classes_np == d
        class_counts[d] = int(mask.sum())
        if mask.sum() > 0:
            class_confidences[d] = float(np.mean(max_probs_np[mask]))
        else:
            class_confidences[d] = 0.0

    class_dist = {d: count / n for d, count in class_counts.items()}

    # Class balance entropy (max = log(10) ≈ 2.302 for uniform over 10 classes)
    freqs = np.array([class_dist[d] for d in range(10)])
    freqs = freqs[freqs > 0]  # avoid log(0)
    entropy = -np.sum(freqs * np.log(freqs))

    return {
        "mean_confidence": mean_confidence,
        "frac_high_confidence": frac_high_conf,
        "confidence_threshold": confidence_threshold,
        "predicted_classes": pred_classes_np,
        "class_distribution": class_dist,
        "class_balance_entropy": entropy,
        "per_class_confidence": class_confidences,
    }


# =====================================================================
# 4.  Batched sampling helpers (faster than one-at-a-time)
# =====================================================================

def _gibbs_sample_batched(model, num_samples: int, burn_in: int,
                          verbose: bool = True) -> torch.Tensor:
    """
    Generate samples via block Gibbs sampling in a single batch.

    Uses gibbs_sample_step_with_coloring which supports batch dimension,
    so we run all chains in parallel (much faster than one-at-a-time).

    Args:
        model: Trained CustomBoltzmannMachine.
        num_samples: Number of samples to generate.
        burn_in: Number of Gibbs steps per chain.
        verbose: Print progress.

    Returns:
        (num_samples, num_visible) tensor.
    """
    model.eval()
    v = torch.bernoulli(torch.full((num_samples, model.num_visible), 0.5, device=device))
    h = torch.bernoulli(torch.full((num_samples, model.num_hidden), 0.5, device=device))

    if verbose:
        print(f"  Batched Gibbs: {num_samples} chains × {burn_in} steps...")

    with torch.no_grad():
        for step in range(burn_in):
            v, h = model.gibbs_sample_step_with_coloring(v, h, track_grad=False)
            if verbose and (step + 1) % max(1, burn_in // 5) == 0:
                print(f"    Step {step+1}/{burn_in}")

    return v.detach()


def _sa_sample_batched(model, num_samples: int, start_temp: float, end_temp: float,
                       max_iterations: int, verbose: bool = True) -> torch.Tensor:
    """
    Generate samples via Simulated Annealing using block Gibbs updates.

    Instead of the original sequential per-unit SA, this uses the block Gibbs
    coloring update at each temperature (equivalent to a full sweep). This is
    dramatically faster for large models while still performing temperature
    annealing.

    At each temperature T, we compute local fields as usual but scale them by
    1/T before applying sigmoid to get sampling probabilities. At T=1 this is
    standard Gibbs; at T>1 it's "hotter" (more random); at T<1 it's "colder"
    (more greedy toward low-energy states).

    Args:
        model: Trained CustomBoltzmannMachine.
        num_samples: Number of samples.
        start_temp: Starting temperature (>0, typically >1).
        end_temp: Final temperature (>0, typically <1).
        max_iterations: Number of temperature steps.
        verbose: Print progress.

    Returns:
        (num_samples, num_visible) tensor of visible samples.
    """
    model.eval()

    temp_schedule = np.logspace(
        np.log10(start_temp), np.log10(end_temp), max_iterations, dtype=np.float64
    )

    v = torch.bernoulli(torch.full((num_samples, model.num_visible), 0.5, device=device))
    h = torch.bernoulli(torch.full((num_samples, model.num_hidden), 0.5, device=device))

    # Track best (lowest energy) state per chain
    best_v = v.clone()
    best_energy = model.energy(v, h)  # (num_samples,)

    if verbose:
        print(f"  Batched SA: {num_samples} chains × {max_iterations} temp steps "
              f"(T: {start_temp:.2f} → {end_temp:.2f})...")

    W_vv, W_hh, W_vh = model._get_masked_weights()
    W_vv = W_vv.detach()
    W_hh = W_hh.detach()
    W_vh = W_vh.detach()

    with torch.no_grad():
        for step, temperature in enumerate(temp_schedule):
            inv_temp = 1.0 / float(temperature)

            # Block Gibbs update at this temperature
            # We iterate over color classes, same as gibbs_sample_step_with_coloring,
            # but scale the local field by inverse temperature.
            for color in range(model.num_colors):
                v_idx = model.color_to_v_indices[color]
                h_idx = model.color_to_h_indices[color]

                if v_idx is not None and len(v_idx) > 0:
                    field_v = (v @ W_vv[:, v_idx] +
                               h @ W_vh.T[:, v_idx] +
                               model.b_v[v_idx])
                    prob_v = torch.sigmoid(inv_temp * field_v)
                    v[:, v_idx] = torch.bernoulli(prob_v)

                if h_idx is not None and len(h_idx) > 0:
                    field_h = (v @ W_vh[:, h_idx] +
                               h @ W_hh[:, h_idx] +
                               model.b_h[h_idx])
                    prob_h = torch.sigmoid(inv_temp * field_h)
                    h[:, h_idx] = torch.bernoulli(prob_h)

            # Track best
            current_energy = model.energy(v, h)
            improved = current_energy < best_energy
            best_v[improved] = v[improved].clone()
            best_energy[improved] = current_energy[improved]

            if verbose and (step + 1) % max(1, max_iterations // 5) == 0:
                mean_e = current_energy.mean().item()
                best_e = best_energy.mean().item()
                print(f"    Step {step+1}/{max_iterations}  T={temperature:.3f}  "
                      f"E_mean={mean_e:.1f}  E_best_mean={best_e:.1f}")

    return best_v.detach()


def _neal_sample(model, num_samples: int, num_sweeps: int = 100,
                 beta_range: tuple[float, float] | None = (0.01, 10.0),
                 seed: int | None = None,
                 verbose: bool = True) -> torch.Tensor:
    """
    num_sweeps=100,
    beta_range=(0.01, 10.0) was good in RBM case. 
    Maybe not in general, but good to start with. 

    Delegates to BM_Neal_Sampler in bolmaqua.  This thin wrapper exists
    so that eval_samples_fullMNIST can call it with a uniform interface.

    Args:
        model: Trained CustomBoltzmannMachine.
        num_samples: Number of SA reads.
        num_sweeps: Number of sweeps per read.
        beta_range: Optional (beta_start, beta_end).
        seed: PRNG seed.
        verbose: Print progress.

    Returns:
        (num_samples, num_visible) tensor on CPU.
    """
    from bolmaqua import BM_Neal_Sampler
    return BM_Neal_Sampler(
        model, num_samples=num_samples, num_sweeps=num_sweeps,
        beta_range=beta_range, seed=seed, verbose=verbose,
    )


# =====================================================================
# 5.  Main evaluation function
# =====================================================================

def eval_samples_fullMNIST(
    model,
    real_data: torch.Tensor,
    grid_shape: tuple[int, int] = (28, 28),
    num_samples: int = 300,
    classifier: MNISTQualityClassifier | None = None,
    classifier_path: str = _DEFAULT_CLASSIFIER_PATH,
    # Gibbs parameters
    gibbs_burn_in: int = 3000,
    # SA parameters (sequential, original implementation)
    sa_start_temp: float = 5.0,
    sa_end_temp: float = 0.5,
    sa_iterations: int = 64,
    # Batched SA parameters (block Gibbs-based, fast)
    batched_sa_start_temp: float = 10.0,
    batched_sa_end_temp: float = 0.2,
    batched_sa_iterations: int = 500,
    # Neal (D-Wave) SA parameters
    neal_num_sweeps: int = 20,
    neal_beta_range: tuple[float, float] | None = (0.1, 5.0),
    neal_seed: int | None = None,
    # Sampling methods to run
    sampling_methods: list[str] | None = None,
    # FID parameters
    n_fid_bootstrap: int = 100,
    fid_ci_level: float = 0.95,
    # Misc
    confidence_threshold: float = 0.8,
    real_features_cache: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """
    Comprehensive sample quality evaluation for a trained BM on full-size MNIST.

    Generates samples via multiple sampling methods, evaluates each with FID
    (+ bootstrap CI) and classifier-based metrics.

    Args:
        model: Trained CustomBoltzmannMachine.
        real_data: (N, 784) tensor of real binarized MNIST images (used as FID reference).
        grid_shape: Image dimensions, default (28, 28).
        num_samples: Number of samples to generate per sampling method.
        classifier: Pre-loaded classifier (if None, loads/trains automatically).
        classifier_path: Path for classifier cache.

        gibbs_burn_in: Burn-in steps for batched Gibbs sampling.

        sa_start_temp, sa_end_temp, sa_iterations: Parameters for the original
            sequential SA sampler (BM_SimAnn_Sampler). Slow but matches original code.

        batched_sa_start_temp, batched_sa_end_temp, batched_sa_iterations:
            Parameters for the fast batched SA sampler. Uses block Gibbs at
            annealed temperatures.

        neal_num_sweeps: Number of sweeps for the Neal SA sampler.
        neal_beta_range: Optional inverse-temperature range for Neal SA.
        neal_seed: PRNG seed for Neal SA reproducibility.

        sampling_methods: List of methods to run. Options:
            - "gibbs": Batched block Gibbs sampling (fast)
            - "sa_batched": Batched SA with block Gibbs updates (fast, recommended)
            - "sa_sequential": Original sequential SA (slow, for comparison)
            - "neal": D-Wave Neal SimulatedAnnealingSampler (closest to real QA)
            Default: ["gibbs", "sa_batched"]

        n_fid_bootstrap: Number of bootstrap resamples for FID CI.
        fid_ci_level: Confidence level for FID CI.
        confidence_threshold: Threshold for "high confidence" classification.
        real_features_cache: Precomputed features for real data (avoids recomputation).
        verbose: Print progress.

    Returns:
        dict with structure:
        {
            'sampling_results': {
                'gibbs': {
                    'samples': Tensor (N, 784),
                    'fid': {...},  # fid, ci_low, ci_high, fid_std
                    'classifier': {...},  # mean_confidence, frac_high_confidence, ...
                    'time_seconds': float,
                },
                'sa_batched': { ... },
                ...
            },
            'best_method': str,  # method name with lowest FID
            'best_fid': float,
            'real_features': np.ndarray,  # cache for reuse
            'classifier': MNISTQualityClassifier,
        }
    """
    if sampling_methods is None:
        sampling_methods = ["gibbs", "sa_batched"]

    valid_methods = {"gibbs", "sa_batched", "sa_sequential", "neal"}
    for m in sampling_methods:
        if m not in valid_methods:
            raise ValueError(f"Unknown sampling method '{m}'. Valid: {valid_methods}")

    # --- Load / train classifier ---
    if classifier is None:
        classifier = train_mnist_classifier(save_path=classifier_path)

    # --- Compute real data features (once) ---
    if real_features_cache is not None:
        real_features = real_features_cache
    else:
        if verbose:
            print("[eval_quality] Computing features for real data...")
        # Use a random subset of real data (10K samples for stable FID reference)
        n_real = min(10000, len(real_data))
        idx = torch.randperm(len(real_data))[:n_real]
        real_subset = real_data[idx]
        real_features = _compute_features(classifier, real_subset)
        if verbose:
            print(f"  Real features: {real_features.shape}")

    # --- Generate and evaluate for each sampling method ---
    results = {}

    for method in sampling_methods:
        if verbose:
            print(f"\n{'='*60}")
            print(f"[eval_quality] Sampling method: {method}")
            print(f"{'='*60}")

        t0 = time.time()

        if method == "gibbs":
            samples = _gibbs_sample_batched(
                model, num_samples, gibbs_burn_in, verbose=verbose)

        elif method == "sa_batched":
            samples = _sa_sample_batched(
                model, num_samples,
                start_temp=batched_sa_start_temp,
                end_temp=batched_sa_end_temp,
                max_iterations=batched_sa_iterations,
                verbose=verbose)

        elif method == "sa_sequential":
            # Use original BM_SimAnn_Sampler from bolmaqua
            from bolmaqua import BM_SimAnn_Sampler
            samples = BM_SimAnn_Sampler(
                model=model,
                start_temp=sa_start_temp,
                end_temp=sa_end_temp,
                max_iterations=sa_iterations,
                num_samples=num_samples,
                track_best=True,
                verbose=verbose)

        elif method == "neal":
            samples = _neal_sample(
                model, num_samples=num_samples,
                num_sweeps=neal_num_sweeps,
                beta_range=neal_beta_range,
                seed=neal_seed,
                verbose=verbose)

        sample_time = time.time() - t0

        if verbose:
            print(f"  Sampling time: {sample_time:.1f}s")

        # Extract features
        gen_features = _compute_features(classifier, samples)

        # FID with CI
        fid_result = compute_fid_with_ci(
            real_features, gen_features,
            n_bootstrap=n_fid_bootstrap,
            ci_level=fid_ci_level,
        )

        # Classifier metrics
        clf_metrics = compute_classifier_metrics(
            classifier, samples,
            confidence_threshold=confidence_threshold,
        )

        if verbose:
            print(f"\n  --- Results for {method} ---")
            print(f"  FID:              {fid_result['fid']:.2f}  "
                  f"({fid_result['ci_level']*100:.0f}% CI: "
                  f"[{fid_result['ci_low']:.2f}, {fid_result['ci_high']:.2f}])")
            print(f"  Mean confidence:  {clf_metrics['mean_confidence']:.3f}")
            print(f"  High confidence:  {clf_metrics['frac_high_confidence']:.1%} "
                  f"(>{confidence_threshold})")
            print(f"  Class balance H:  {clf_metrics['class_balance_entropy']:.3f} "
                  f"(max={np.log(10):.3f})")
            print(f"  Class distribution: ", end="")
            for d in range(10):
                pct = clf_metrics['class_distribution'][d] * 100
                if pct > 0:
                    print(f"{d}:{pct:.0f}% ", end="")
            print()

        results[method] = {
            "samples": samples.cpu(),
            "fid": fid_result,
            "classifier": clf_metrics,
            "time_seconds": sample_time,
        }

    # --- Determine best method ---
    best_method = min(results, key=lambda m: results[m]["fid"]["fid"])
    best_fid = results[best_method]["fid"]["fid"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"[eval_quality] SUMMARY")
        print(f"{'='*60}")
        for m in results:
            r = results[m]
            fid_str = (f"FID={r['fid']['fid']:.2f} "
                       f"[{r['fid']['ci_low']:.2f}, {r['fid']['ci_high']:.2f}]")
            conf_str = f"conf={r['classifier']['mean_confidence']:.3f}"
            hc_str = f"HC={r['classifier']['frac_high_confidence']:.1%}"
            t_str = f"t={r['time_seconds']:.1f}s"
            best_marker = " ★" if m == best_method else ""
            print(f"  {m:20s}  {fid_str}  {conf_str}  {hc_str}  {t_str}{best_marker}")
        print()

    return {
        "sampling_results": results,
        "best_method": best_method,
        "best_fid": best_fid,
        "real_features": real_features,
        "classifier": classifier,
    }


# =====================================================================
# 6.  Quick standalone test
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("eval_quality.py — Standalone classifier training test")
    print("=" * 60)
    clf = train_mnist_classifier(force_retrain=True)
    print(f"\nClassifier ready on {device}.")
    print(f"Feature dim: {clf.feature_dim}")

    # Quick sanity check: FID calibration
    print("\nSanity check: FID calibration with PCA-32 (default), N_gen=400...")
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    imgs = (ds.data.float() / 255.0 >= 0.5).float().view(-1, 784)
    feat_a = _compute_features(clf, imgs[:10000])

    feat_b = _compute_features(clf, imgs[10000:10400])
    result = compute_fid_with_ci(feat_a, feat_b, n_bootstrap=50, pca_components=32)
    print(f"  FID(real, real)  = {result['fid']:.2f}  "
          f"CI=[{result['ci_low']:.2f}, {result['ci_high']:.2f}]  (bias floor)")

    # Noise vs real (should be very high)
    import torch as _torch
    noise = _torch.bernoulli(_torch.full((400, 784), 0.5))
    feat_noise = _compute_features(clf, noise)
    result_noise = compute_fid_with_ci(feat_a, feat_noise, n_bootstrap=50, pca_components=32)
    print(f"  FID(real, noise) = {result_noise['fid']:.2f}  "
          f"CI=[{result_noise['ci_low']:.2f}, {result_noise['ci_high']:.2f}]  (noise ceiling)")
    print(f"  Dynamic range: {result_noise['fid']/max(result['fid'], 0.01):.0f}x")
    print("\nDone.")
