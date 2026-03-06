"""
classifier_hyperparam_search.py — Find the best MNIST classifier for FID evaluation

The classifier in eval_quality.py is used to extract features for FID (Fréchet
Inception Distance) computation. The quality of those features directly affects
how well FID can discriminate between good and bad generative samples.

This script searches over classifier architectures and training hyperparameters
to find a model that produces the best feature space for FID, evaluated by:

  1. FID Dynamic Range — ratio of FID(real, noise) to FID(real, real_holdout).
     Higher = better separation between good and bad samples.
  2. Mode Sensitivity — does FID increase when we drop half the digit classes?
     Tests detection of mode collapse.
  3. Degradation Sensitivity — does FID increase when we corrupt real images
     with random bit flips? Tests detection of noisy/blurry samples.
  4. FID Stability — low bootstrap variance on FID(real, real_holdout).
  5. Test Accuracy — classification accuracy (features must capture digit structure).

A composite score ranks candidates. The best model's config and weights are
saved so it can replace the current classifier in eval_quality.py.

Usage:
    python classifier_hyperparam_search.py --trials 50 --hours 2
    python classifier_hyperparam_search.py --trials 100 --hours 6 --top-n 10
"""

import os
import sys
import time
import json
import argparse
import warnings
import copy
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import datasets, transforms
from scipy import linalg

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# =====================================================================
# 1.  Flexible Classifier Architecture
# =====================================================================

class FlexibleMNISTClassifier(nn.Module):
    """
    Parameterized CNN for binarized 28x28 MNIST (10 classes).

    The architecture is controlled by a config dict with keys:
        n_conv_layers: int (2-5)
        base_channels: int (8, 16, 24, 32, 48)
        channel_growth: float (1.5, 2.0, 2.5, 3.0)
        kernel_size: int (3, 5)
        use_batchnorm: bool
        dropout_rate: float (0.0 - 0.4)
        pool_size: int (1, 2, 3) — AdaptiveAvgPool output spatial size
        activation: str ('relu', 'leaky_relu', 'elu')
        use_fc_hidden: bool — add a hidden FC layer before the classifier head
        fc_hidden_dim: int (64, 128, 256) — size of hidden FC (if used)
        fc_dropout: float (0.0 - 0.5) — dropout before classifier head

    The feature vector is the output after the last conv block + pool + flatten
    (+ optional FC hidden layer), before the final classification linear layer.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        n_layers = config['n_conv_layers']
        base_ch = config['base_channels']
        growth = config['channel_growth']
        ks = config['kernel_size']
        use_bn = config['use_batchnorm']
        dropout = config['dropout_rate']
        pool_size = config['pool_size']
        activation = config.get('activation', 'relu')

        # Build activation function
        act_fn = {
            'relu': nn.ReLU,
            'leaky_relu': lambda: nn.LeakyReLU(0.1),
            'elu': nn.ELU,
        }[activation]

        # Build conv layers
        layers = []
        in_ch = 1
        for i in range(n_layers):
            out_ch = int(base_ch * (growth ** i))
            out_ch = max(out_ch, 4)  # minimum 4 channels
            layers.append(nn.Conv2d(in_ch, out_ch, ks, stride=2, padding=ks // 2))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch

        self.convs = nn.Sequential(*layers)

        # Compute spatial size after convolutions (each stride-2 conv halves,
        # with padding=ks//2 the formula is ceil(dim/2) per layer)
        spatial = 28
        for _ in range(n_layers):
            spatial = (spatial + 1) // 2  # ceil(spatial / 2) due to padding
        # Clamp pool_size to not exceed the spatial dimension
        pool_size = min(pool_size, spatial)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        # Compute feature dim from conv output
        conv_feat_dim = out_ch * pool_size * pool_size

        # Optional hidden FC layer
        if config.get('use_fc_hidden', False):
            fc_hidden = config.get('fc_hidden_dim', 128)
            fc_drop = config.get('fc_dropout', 0.0)
            fc_layers = [nn.Linear(conv_feat_dim, fc_hidden), act_fn()]
            if fc_drop > 0:
                fc_layers.append(nn.Dropout(fc_drop))
            self.fc_hidden = nn.Sequential(*fc_layers)
            self.feature_dim = fc_hidden
        else:
            self.fc_hidden = None
            fc_drop = config.get('fc_dropout', 0.0)
            self.feature_dim = conv_feat_dim

        # Dropout before classifier head (even without fc_hidden)
        self.head_dropout = nn.Dropout(fc_drop) if fc_drop > 0 and self.fc_hidden is None else None

        # Classification head
        self.fc = nn.Linear(self.feature_dim, 10)

    def features(self, x):
        """Return the feature vector (before the classification head)."""
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.convs(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.fc_hidden is not None:
            x = self.fc_hidden(x)
        elif self.head_dropout is not None:
            x = self.head_dropout(x)
        return x

    def forward(self, x):
        feat = self.features(x)
        return self.fc(feat)

    def predict_proba(self, x):
        """Return softmax probabilities (B, 10)."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# 2.  Search Space
# =====================================================================

def sample_config(rng: np.random.RandomState) -> dict:
    """Sample a random classifier configuration."""
    config = {}

    # Architecture (cast numpy types to native Python for torch/json compat)
    config['n_conv_layers'] = int(rng.choice([2, 3, 4, 5]))
    config['base_channels'] = int(rng.choice([8, 12, 16, 24, 32, 48]))
    config['channel_growth'] = float(rng.choice([1.5, 2.0, 2.5, 3.0]))
    config['kernel_size'] = int(rng.choice([3, 5]))
    config['use_batchnorm'] = bool(rng.choice([True, False]))
    config['dropout_rate'] = float(rng.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3]))
    config['pool_size'] = int(rng.choice([1, 2, 3]))
    config['activation'] = str(rng.choice(['relu', 'leaky_relu', 'elu']))

    # Optional FC hidden layer
    config['use_fc_hidden'] = bool(rng.choice([True, False]))
    if config['use_fc_hidden']:
        config['fc_hidden_dim'] = int(rng.choice([64, 96, 128, 192, 256, 384]))
        config['fc_dropout'] = float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.5]))
    else:
        config['fc_dropout'] = float(rng.choice([0.0, 0.1, 0.2, 0.3]))

    # Training
    config['lr'] = float(10 ** rng.uniform(-4, -1.5))  # ~1e-4 to ~3e-2
    config['weight_decay'] = float(rng.choice([0.0]) if rng.rand() < 0.2
                                   else 10 ** rng.uniform(-6, -2.5))
    config['batch_size'] = int(rng.choice([64, 128, 256, 512]))
    config['epochs'] = int(rng.choice([10, 15, 20, 25, 30, 40]))
    config['optimizer'] = str(rng.choice(['adam', 'adamw']))
    config['scheduler'] = str(rng.choice(['step', 'cosine', 'none']))
    config['label_smoothing'] = float(rng.choice([0.0, 0.0, 0.05, 0.1]))

    return config


def get_current_default_config() -> dict:
    """Return the config matching the current eval_quality.py classifier."""
    return {
        'n_conv_layers': 3,
        'base_channels': 16,
        'channel_growth': 2.0,
        'kernel_size': 3,
        'use_batchnorm': False,
        'dropout_rate': 0.0,
        'pool_size': 2,
        'activation': 'relu',
        'use_fc_hidden': False,
        'fc_dropout': 0.0,
        'lr': 1e-3,
        'weight_decay': 0.0,
        'batch_size': 256,
        'epochs': 12,
        'optimizer': 'adam',
        'scheduler': 'step',
        'label_smoothing': 0.0,
    }


# =====================================================================
# 3.  Data Preparation
# =====================================================================

def prepare_data(binarize_threshold: float = 0.5, val_size: int = 10000):
    """
    Load MNIST, binarize, and split into train/val/test.

    Returns:
        train_imgs: (N_train, 1, 28, 28) binarized
        train_labels: (N_train,)
        val_imgs: (N_val, 1, 28, 28) binarized
        val_labels: (N_val,)
        test_imgs: (N_test, 1, 28, 28) binarized
        test_labels: (N_test,)
    """
    train_ds = datasets.MNIST(root="./data", train=True, download=True,
                              transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root="./data", train=False, download=True,
                             transform=transforms.ToTensor())

    def binarize(ds):
        imgs = ds.data.float() / 255.0
        imgs = (imgs >= binarize_threshold).float()
        imgs = imgs.unsqueeze(1)  # (N, 1, 28, 28)
        return imgs, ds.targets

    all_train_imgs, all_train_labels = binarize(train_ds)
    test_imgs, test_labels = binarize(test_ds)

    # Split training into train + validation
    n = len(all_train_imgs)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_imgs = all_train_imgs[train_idx]
    train_labels = all_train_labels[train_idx]
    val_imgs = all_train_imgs[val_idx]
    val_labels = all_train_labels[val_idx]

    return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels


# =====================================================================
# 4.  Training
# =====================================================================

def train_classifier(config: dict, train_imgs, train_labels, val_imgs, val_labels,
                     patience: int = 7, verbose: bool = False) -> tuple:
    """
    Train a FlexibleMNISTClassifier with early stopping on val accuracy.

    Returns:
        (model, train_history) where train_history is a dict with
        'train_acc', 'val_acc', 'train_loss', 'val_loss', 'best_epoch', 'stopped_early'
    """
    model = FlexibleMNISTClassifier(config).to(device)

    # Check parameter count — skip if unreasonably large
    n_params = model.count_parameters()
    if n_params > 5_000_000:
        raise ValueError(f"Model too large: {n_params:,} parameters")

    bs = config['batch_size']
    train_loader = DataLoader(
        TensorDataset(train_imgs, train_labels),
        batch_size=bs, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_imgs, val_labels),
        batch_size=512, shuffle=False,
    )

    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                     weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                      weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    # Scheduler
    if config['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'])
    else:
        scheduler = None

    # Label smoothing
    ls = config.get('label_smoothing', 0.0)

    history = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': [],
        'best_epoch': 0, 'stopped_early': False,
    }

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config['epochs']):
        # --- Train ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels, label_smoothing=ls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        if scheduler is not None:
            scheduler.step()

        train_acc = correct / total
        train_loss = total_loss / total
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

        # --- Validate ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss_sum / val_total
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"    Epoch {epoch+1}/{config['epochs']}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                  f"val_loss={val_loss:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            history['best_epoch'] = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                history['stopped_early'] = True
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1} "
                          f"(best val_acc={best_val_acc:.4f} at epoch {history['best_epoch']})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, history


# =====================================================================
# 5.  Feature Extraction & FID
# =====================================================================

def extract_features(model, images: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    """Extract feature vectors from a model that has a .features() method."""
    model.eval()
    all_feats = []
    images = images.to(device)
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            feats = model.features(batch)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def pca_reduce(real_features: np.ndarray, other_features: np.ndarray,
               n_components: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """PCA reduction (fit on real, apply to both). Same as eval_quality._pca_reduce."""
    n_components = min(n_components, real_features.shape[1], real_features.shape[0])
    mu = np.mean(real_features, axis=0)
    real_c = real_features - mu
    other_c = other_features - mu
    U, S, Vt = np.linalg.svd(real_c, full_matrices=False)
    components = Vt[:n_components]
    return real_c @ components.T, other_c @ components.T


def compute_fid(feat_a: np.ndarray, feat_b: np.ndarray, eps: float = 1e-6) -> float:
    """Compute FID between two feature arrays. Same as eval_quality.compute_fid."""
    mu_a = np.mean(feat_a, axis=0)
    mu_b = np.mean(feat_b, axis=0)
    sigma_a = np.cov(feat_a, rowvar=False)
    sigma_b = np.cov(feat_b, rowvar=False)
    sigma_a += np.eye(sigma_a.shape[0]) * eps
    sigma_b += np.eye(sigma_b.shape[0]) * eps

    diff = mu_a - mu_b
    mean_term = diff @ diff

    product = sigma_a @ sigma_b
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covmean_sq, _ = linalg.sqrtm(product, disp=False)

    if np.iscomplexobj(covmean_sq):
        if np.max(np.abs(covmean_sq.imag)) < 1e-3 * np.max(np.abs(covmean_sq.real)):
            covmean_sq = covmean_sq.real
        else:
            offset = np.eye(sigma_a.shape[0]) * eps * 10
            covmean_sq, _ = linalg.sqrtm((sigma_a + offset) @ (sigma_b + offset), disp=False)
            covmean_sq = covmean_sq.real

    trace_term = np.trace(sigma_a + sigma_b - 2.0 * covmean_sq)
    return max(float(mean_term + trace_term), 0.0)


# =====================================================================
# 6.  FID Calibration Evaluation
# =====================================================================

def evaluate_fid_quality(
    model,
    test_imgs: torch.Tensor,
    test_labels: torch.Tensor,
    pca_components: int = 32,
    n_eval_samples: int = 500,
    n_bootstrap: int = 50,
    rng: np.random.RandomState = None,
) -> dict:
    """
    Evaluate how good this model's feature space is for FID computation.

    Tests:
      1. FID baseline: FID(real_ref, real_holdout) — the bias floor
      2. FID noise ceiling: FID(real_ref, pure_noise) — upper bound
      3. Dynamic range: noise_fid / baseline_fid
      4. Mode sensitivity: FID(real_ref, digits_0_to_4_only) — detects mode drop
      5. Degradation sensitivity: FID(real_ref, real_with_10%_bitflips)
      6. Stability: bootstrap std on baseline FID

    All FIDs use PCA reduction to match the production pipeline.

    Returns:
        dict with all metrics + composite score
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Extract features for all test data
    all_features = extract_features(model, test_imgs)
    n_test = len(all_features)

    # Split test features into reference (first half) and holdout (second half)
    perm = rng.permutation(n_test)
    ref_idx = perm[:n_test // 2]
    holdout_idx = perm[n_test // 2:]

    ref_features = all_features[ref_idx]

    # Take n_eval_samples from holdout for consistent comparison
    holdout_features = all_features[holdout_idx[:n_eval_samples]]

    # --- 1. Baseline FID: real vs real ---
    ref_pca, holdout_pca = pca_reduce(ref_features, holdout_features, pca_components)
    fid_baseline = compute_fid(ref_pca, holdout_pca)

    # Bootstrap for stability
    bootstrap_fids = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(holdout_pca), size=len(holdout_pca), replace=True)
        fid_boot = compute_fid(ref_pca, holdout_pca[idx])
        bootstrap_fids.append(fid_boot)
    fid_std = float(np.std(bootstrap_fids))
    fid_cv = fid_std / max(fid_baseline, 1e-6)  # coefficient of variation

    # --- 2. Noise ceiling FID ---
    noise_imgs = torch.bernoulli(torch.full((n_eval_samples, 1, 28, 28), 0.5))
    noise_features = extract_features(model, noise_imgs)
    _, noise_pca = pca_reduce(ref_features, noise_features, pca_components)
    fid_noise = compute_fid(ref_pca, noise_pca)

    # --- 3. Dynamic range ---
    dynamic_range = fid_noise / max(fid_baseline, 1e-6)

    # --- 4. Mode sensitivity: only digits 0-4 ---
    labels_np = test_labels.numpy() if isinstance(test_labels, torch.Tensor) else test_labels
    mode_mask = np.isin(labels_np, [0, 1, 2, 3, 4])
    mode_idx = np.where(mode_mask)[0]
    if len(mode_idx) >= n_eval_samples:
        mode_subset_idx = rng.choice(mode_idx, size=n_eval_samples, replace=False)
    else:
        mode_subset_idx = mode_idx
    mode_features = all_features[mode_subset_idx]
    _, mode_pca = pca_reduce(ref_features, mode_features, pca_components)
    fid_mode_dropped = compute_fid(ref_pca, mode_pca)
    mode_sensitivity = fid_mode_dropped / max(fid_baseline, 1e-6)

    # --- 5. Degradation sensitivity: flip 10% of pixels ---
    degraded_imgs = test_imgs[holdout_idx[:n_eval_samples]].clone()
    flip_mask = torch.bernoulli(torch.full_like(degraded_imgs, 0.10))
    degraded_imgs = torch.where(flip_mask.bool(), 1.0 - degraded_imgs, degraded_imgs)
    degraded_features = extract_features(model, degraded_imgs)
    _, degraded_pca = pca_reduce(ref_features, degraded_features, pca_components)
    fid_degraded = compute_fid(ref_pca, degraded_pca)
    degradation_sensitivity = fid_degraded / max(fid_baseline, 1e-6)

    # --- 6. Graduated degradation: 5%, 10%, 20%, 30% bit flips ---
    #     We check that FID increases monotonically (feature space is ordered)
    graduated_fids = {}
    for flip_rate in [0.05, 0.10, 0.20, 0.30]:
        deg = test_imgs[holdout_idx[:n_eval_samples]].clone()
        fm = torch.bernoulli(torch.full_like(deg, flip_rate))
        deg = torch.where(fm.bool(), 1.0 - deg, deg)
        deg_feat = extract_features(model, deg)
        _, deg_pca = pca_reduce(ref_features, deg_feat, pca_components)
        graduated_fids[flip_rate] = compute_fid(ref_pca, deg_pca)

    # Check monotonicity
    fid_vals = [graduated_fids[r] for r in sorted(graduated_fids.keys())]
    monotonic_pairs = sum(1 for i in range(len(fid_vals)-1) if fid_vals[i+1] > fid_vals[i])
    monotonicity_score = monotonic_pairs / max(len(fid_vals) - 1, 1)  # 1.0 = perfectly monotonic

    # --- Composite score ---
    # Components (all "higher is better"):
    #   log10(dynamic_range): ~1.0 to 3.0 for typical models
    #   test_accuracy: injected externally, weighted separately
    #   log10(mode_sensitivity): ~0 to 2.0
    #   log10(degradation_sensitivity): ~0 to 1.5
    #   stability: 1/(1 + cv), in (0, 1]
    #   monotonicity: 0 to 1

    stability = 1.0 / (1.0 + fid_cv)

    log_dr = math.log10(max(dynamic_range, 1.0))
    log_ms = math.log10(max(mode_sensitivity, 1.0))
    log_ds = math.log10(max(degradation_sensitivity, 1.0))

    # Composite (test_accuracy added externally in the main loop)
    fid_quality_score = (
        3.0 * log_dr +           # most important: distinguish good from noise
        2.0 * log_ms +           # important: detect mode collapse
        1.5 * log_ds +           # important: detect sample degradation
        1.0 * stability +        # desirable: low noise in measurement
        1.0 * monotonicity_score # desirable: ordered feature space
    )

    return {
        'fid_baseline': fid_baseline,
        'fid_noise': fid_noise,
        'fid_mode_dropped': fid_mode_dropped,
        'fid_degraded': fid_degraded,
        'dynamic_range': dynamic_range,
        'mode_sensitivity': mode_sensitivity,
        'degradation_sensitivity': degradation_sensitivity,
        'fid_std': fid_std,
        'fid_cv': fid_cv,
        'stability': stability,
        'graduated_fids': {str(k): v for k, v in graduated_fids.items()},
        'monotonicity_score': monotonicity_score,
        'fid_quality_score': fid_quality_score,
        'pca_components': pca_components,
        'n_eval_samples': n_eval_samples,
    }


# =====================================================================
# 7.  Test Set Accuracy
# =====================================================================

def evaluate_test_accuracy(model, test_imgs, test_labels, batch_size=512) -> float:
    """Compute classification accuracy on the test set."""
    model.eval()
    loader = DataLoader(TensorDataset(test_imgs, test_labels),
                        batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return correct / total


# =====================================================================
# 8.  Leaderboard Management
# =====================================================================

def update_leaderboard(leaderboard: list, entry: dict, top_n: int) -> list:
    """Insert entry into sorted leaderboard (by composite_score, descending)."""
    leaderboard.append(entry)
    leaderboard.sort(key=lambda x: x['composite_score'], reverse=True)
    return leaderboard[:top_n]


def save_leaderboard(leaderboard: list, path: str):
    """Atomically save leaderboard to JSON."""
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(leaderboard, f, indent=2, default=str)
    os.replace(tmp, path)


def print_leaderboard(leaderboard: list):
    """Print a summary table of the leaderboard."""
    print(f"\n{'='*100}")
    print(f"{'Rank':<5} {'Trial':<7} {'Score':>7} {'Acc':>7} {'FID_DR':>8} "
          f"{'ModeSens':>8} {'DegSens':>8} {'Stab':>6} {'Mono':>5} "
          f"{'FeatDim':>8} {'Params':>10} {'Arch Summary'}")
    print(f"{'-'*100}")
    for rank, entry in enumerate(leaderboard, 1):
        fid_q = entry['fid_quality']
        config = entry['config']
        arch = (f"{config['n_conv_layers']}L "
                f"ch{config['base_channels']}×{config['channel_growth']:.1f} "
                f"k{config['kernel_size']} "
                f"{'BN ' if config['use_batchnorm'] else ''}"
                f"p{config['pool_size']} "
                f"{'FC' + str(config.get('fc_hidden_dim', '')) + ' ' if config.get('use_fc_hidden') else ''}"
                f"{config['activation']}")
        print(f"{rank:<5} {entry['trial']:<7} "
              f"{entry['composite_score']:>7.3f} "
              f"{entry['test_accuracy']:>6.4f} "
              f"{fid_q['dynamic_range']:>8.1f} "
              f"{fid_q['mode_sensitivity']:>8.2f} "
              f"{fid_q['degradation_sensitivity']:>8.2f} "
              f"{fid_q['stability']:>6.3f} "
              f"{fid_q['monotonicity_score']:>5.2f} "
              f"{entry['feature_dim']:>8} "
              f"{entry['n_params']:>10,} "
              f"{arch}")
    print(f"{'='*100}")


# =====================================================================
# 9.  Main Search Loop
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for MNIST quality classifier (FID evaluation)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Maximum number of trials (default: 50)")
    parser.add_argument("--hours", type=float, default=2.0,
                        help="Time budget in hours (default: 2.0)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Track top-N results (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--pca-components", type=int, default=32,
                        help="PCA components for FID (default: 32, matches eval_quality.py)")
    parser.add_argument("--eval-samples", type=int, default=500,
                        help="Samples per FID evaluation (default: 500)")
    parser.add_argument("--include-current", action="store_true", default=True,
                        help="Include current eval_quality.py config as trial 0 (default: True)")
    parser.add_argument("--verbose-training", action="store_true", default=False,
                        help="Print per-epoch training details")
    parser.add_argument("--save-best-model", action="store_true", default=True,
                        help="Save the best model weights (default: True)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(DATA_DIR, f"{timestamp}_classifier_search_results.json")
    # Best model overwrites the standard classifier path used by eval_quality.py
    best_model_path = os.path.join(DATA_DIR, "mnist_quality_classifier.pt")

    print("=" * 80)
    print("Classifier Hyperparameter Search for FID Evaluation")
    print("=" * 80)
    print(f"  Device:         {device}")
    print(f"  Max trials:     {args.trials}")
    print(f"  Time budget:    {args.hours:.1f} hours")
    print(f"  Top-N:          {args.top_n}")
    print(f"  PCA components: {args.pca_components}")
    print(f"  Eval samples:   {args.eval_samples}")
    print(f"  Seed:           {args.seed}")
    print(f"  Results:        {results_path}")
    print()

    rng = np.random.RandomState(args.seed)

    # --- Load data ---
    print("Loading and preparing MNIST data...")
    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = prepare_data()
    print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    # --- Build trial configs ---
    configs = []
    if args.include_current:
        configs.append(("current_default", get_current_default_config()))

    for i in range(args.trials - len(configs)):
        configs.append((f"random_{i+1}", sample_config(rng)))

    # --- Search ---
    leaderboard = []
    start_time = time.time()
    deadline = start_time + args.hours * 3600
    best_model_state = None
    best_model_config = None
    n_completed = 0
    n_failed = 0

    for trial_idx, (trial_name, config) in enumerate(configs):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 0:
            print(f"\n[Time budget exhausted after {elapsed/3600:.2f} hours]")
            break

        print(f"\n{'─'*80}")
        print(f"Trial {trial_idx+1}/{len(configs)}  [{trial_name}]  "
              f"({elapsed/60:.1f}min elapsed, {remaining/60:.1f}min remaining)")
        print(f"{'─'*80}")

        # Print config summary
        arch_str = (f"  Arch: {config['n_conv_layers']}L, ch={config['base_channels']}×"
                    f"{config['channel_growth']:.1f}, k={config['kernel_size']}, "
                    f"{'BN' if config['use_batchnorm'] else 'no-BN'}, "
                    f"drop={config['dropout_rate']}, pool={config['pool_size']}, "
                    f"act={config['activation']}")
        if config.get('use_fc_hidden'):
            arch_str += f", FC={config['fc_hidden_dim']}(drop={config['fc_dropout']})"
        print(arch_str)
        print(f"  Train: lr={config['lr']:.1e}, wd={config['weight_decay']:.1e}, "
              f"bs={config['batch_size']}, epochs={config['epochs']}, "
              f"opt={config['optimizer']}, sched={config['scheduler']}, "
              f"ls={config.get('label_smoothing', 0.0)}")

        try:
            # Build model to check feasibility
            test_model = FlexibleMNISTClassifier(config).to(device)
            n_params = test_model.count_parameters()
            feat_dim = test_model.feature_dim
            print(f"  Params: {n_params:,}  Feature dim: {feat_dim}")
            del test_model

            # Train
            t0 = time.time()
            model, history = train_classifier(
                config, train_imgs, train_labels, val_imgs, val_labels,
                patience=7, verbose=args.verbose_training)
            train_time = time.time() - t0

            best_val_acc = max(history['val_acc'])
            print(f"  Training: {train_time:.1f}s, best_val_acc={best_val_acc:.4f} "
                  f"(epoch {history['best_epoch']}"
                  f"{', early stopped' if history['stopped_early'] else ''})")

            # Skip if accuracy is terrible (features won't be useful)
            if best_val_acc < 0.90:
                print(f"  SKIPPED: val_acc < 0.90 (features likely uninformative)")
                n_failed += 1
                continue

            # Test accuracy
            test_acc = evaluate_test_accuracy(model, test_imgs, test_labels)
            print(f"  Test accuracy: {test_acc:.4f}")

            # FID quality evaluation
            t1 = time.time()
            fid_quality = evaluate_fid_quality(
                model, test_imgs, test_labels,
                pca_components=args.pca_components,
                n_eval_samples=args.eval_samples,
                rng=np.random.RandomState(args.seed + trial_idx),
            )
            eval_time = time.time() - t1

            # Composite score: FID quality + accuracy bonus
            # Accuracy above 0.97 gets diminishing returns
            acc_bonus = 5.0 * min(test_acc, 0.99)
            composite_score = fid_quality['fid_quality_score'] + acc_bonus

            print(f"  FID eval: {eval_time:.1f}s")
            print(f"    Baseline FID:   {fid_quality['fid_baseline']:.2f} "
                  f"(std={fid_quality['fid_std']:.2f})")
            print(f"    Noise FID:      {fid_quality['fid_noise']:.2f}")
            print(f"    Dynamic range:  {fid_quality['dynamic_range']:.1f}×")
            print(f"    Mode sens:      {fid_quality['mode_sensitivity']:.2f}× "
                  f"(FID={fid_quality['fid_mode_dropped']:.2f})")
            print(f"    Degrad sens:    {fid_quality['degradation_sensitivity']:.2f}× "
                  f"(FID={fid_quality['fid_degraded']:.2f})")
            print(f"    Monotonicity:   {fid_quality['monotonicity_score']:.2f}")
            print(f"    Stability:      {fid_quality['stability']:.3f}")
            grad_str = "    Graduated FIDs: "
            for k, v in sorted(fid_quality['graduated_fids'].items()):
                grad_str += f"{k}→{v:.1f}  "
            print(grad_str)
            print(f"  ► COMPOSITE SCORE: {composite_score:.3f}")

            # Build leaderboard entry (must be JSON-serializable)
            entry = {
                'trial': trial_idx + 1,
                'trial_name': trial_name,
                'composite_score': composite_score,
                'test_accuracy': test_acc,
                'best_val_accuracy': best_val_acc,
                'fid_quality': fid_quality,
                'config': config,
                'feature_dim': feat_dim,
                'n_params': n_params,
                'train_time_s': train_time,
                'eval_time_s': eval_time,
                'best_epoch': history['best_epoch'],
                'stopped_early': history['stopped_early'],
            }

            old_best = leaderboard[0]['composite_score'] if leaderboard else -float('inf')
            leaderboard = update_leaderboard(leaderboard, entry, args.top_n)
            save_leaderboard(leaderboard, results_path)

            if composite_score > old_best:
                print(f"  ★ NEW BEST! (previous best: {old_best:.3f})")
                best_model_state = copy.deepcopy(model.state_dict())
                best_model_config = copy.deepcopy(config)
                # Save best model weights
                if args.save_best_model:
                    torch.save({
                        'model_state_dict': best_model_state,
                        'config': best_model_config,
                        'test_accuracy': test_acc,
                        'composite_score': composite_score,
                        'fid_quality': fid_quality,
                        'feature_dim': feat_dim,
                        'n_params': n_params,
                    }, best_model_path)
                    print(f"  Saved best model to {best_model_path}")

            n_completed += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            n_failed += 1
            continue

    # --- Final summary ---
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"  Total time:     {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"  Trials:         {n_completed} completed, {n_failed} failed")
    print(f"  Results saved:  {results_path}")
    if best_model_state is not None:
        print(f"  Best model:     {best_model_path} (overwrites eval_quality.py classifier)")

    if leaderboard:
        print_leaderboard(leaderboard)

        # Print the best config in detail
        best = leaderboard[0]
        print(f"\nBest configuration (trial {best['trial']}, "
              f"score={best['composite_score']:.3f}):")
        print(f"  Test accuracy:        {best['test_accuracy']:.4f}")
        print(f"  Feature dim:          {best['feature_dim']}")
        print(f"  Parameters:           {best['n_params']:,}")
        print(f"  FID dynamic range:    {best['fid_quality']['dynamic_range']:.1f}×")
        print(f"  Mode sensitivity:     {best['fid_quality']['mode_sensitivity']:.2f}×")
        print(f"  Degrad sensitivity:   {best['fid_quality']['degradation_sensitivity']:.2f}×")
        print()
        print("  Config dict:")
        for k, v in sorted(best['config'].items()):
            print(f"    {k}: {v}")

        print(f"\nTo use this model in eval_quality.py, update MNISTQualityClassifier")
        print(f"to match this architecture, or load via FlexibleMNISTClassifier(config).")


if __name__ == "__main__":
    main()
