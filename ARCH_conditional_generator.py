"""
Conditional Generative Boltzmann Machine

Instead of learning p(pixels), learns p(pixels | label).
Enables controlled sampling: "generate a specific digit".

Training: Labels are visible units (clamped to true labels)
Sampling: Clamp desired label, sample pixels
"""

import torch
import numpy as np
from bolmaqua import device

def add_label_nodes_for_conditional(G, visible_nodes, hidden_nodes, num_classes=10):
    """
    Add label nodes for conditional generation.
    Similar to classification, but different usage during sampling.
    
    Returns:
        G_extended: Graph with label nodes
        label_nodes: List of label node IDs [node_for_0, node_for_1, ..., node_for_9]
        node_labels_extended: Updated dict
    """
    # Reuse the classification infrastructure
    from classifierhelper import add_label_nodes_to_graph
    
    G_extended, label_node_groups, node_labels_extended = add_label_nodes_to_graph(
        G, visible_nodes, hidden_nodes, 
        num_classes=num_classes, 
        nodes_per_label=1  # Always 1 for conditional generation
    )
    
    # Flatten label_node_groups (since nodes_per_label=1)
    label_nodes = [group[0] for group in label_node_groups]
    
    return G_extended, label_nodes, node_labels_extended


def prepare_conditional_batch(pixels, labels, label_nodes, num_classes=10):
    """
    Prepare batch for conditional generative training.
    Same as classification training: labels are one-hot encoded.
    """
    from classifierhelper import prepare_classification_batch
    
    # Use classification function (it does one-hot encoding)
    extended_visible = prepare_classification_batch(
        pixels, labels, 
        [[ln] for ln in label_nodes],  # Wrap each in list for compatibility
        num_classes=num_classes,
        nodes_per_label=1
    )
    
    return extended_visible


def train_conditional_bm(model, data_loader, optimizer, num_epochs, k_steps, 
                        label_nodes, batch_size, step_size, num_classes=10):
    """
    Train conditional generative BM.
    Identical to discriminative training, but we'll use it differently at test time.
    """
    from bolmaqua import compute_pseudolikelihood
    
    loss_history = []
    pll_values = []
    
    print(f"\nTraining Conditional Generative BM (epochs={num_epochs}, k_steps={k_steps})...")
    print(f"  Model learns: p(pixels | label)")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (pixels, labels) in enumerate(data_loader):
            pixels = pixels.to(device)
            labels = labels.to(device)
            
            # Extend visible to include labels
            extended_visible = prepare_conditional_batch(pixels, labels, label_nodes, num_classes)
            
            optimizer.zero_grad()
            loss, _ = model(extended_visible, k_steps=k_steps)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # PLL evaluation
        with torch.no_grad():
            sample_batch = next(iter(data_loader))
            sample_pixels, sample_labels = sample_batch[0].to(device), sample_batch[1].to(device)
            sample_extended = prepare_conditional_batch(sample_pixels, sample_labels, label_nodes, num_classes)
            pll = compute_pseudolikelihood(model, sample_extended, num_samples=50)
            pll_values.append(pll)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | PLL: {pll:.4f}")
    
    return {'pcd_loss': loss_history, 'pll': pll_values}


def sample_conditional(model, label_nodes, target_labels, num_pixels, 
                      burn_in_steps=1000, method='gibbs'):
    """
    Sample from conditional BM: generate images for specific digits.
    
    Args:
        model: Trained conditional BM
        label_nodes: List of label node IDs
        target_labels: List/tensor of desired labels (e.g., [0, 1, 7, 7, 3])
        num_pixels: Number of pixel units (784 for 28x28)
        burn_in_steps: Gibbs burn-in steps
        method: 'gibbs' (only Gibbs supported for now)
    
    Returns:
        samples: (num_samples, num_pixels) tensor of generated images
    """
    model.eval()
    
    if isinstance(target_labels, int):
        target_labels = [target_labels]
    
    num_samples = len(target_labels)
    num_classes = len(label_nodes)
    
    print(f"\nGenerating {num_samples} conditional samples...")
    print(f"  Target labels: {target_labels}")
    print(f"  Burn-in: {burn_in_steps} steps")
    
    with torch.no_grad():
        # Initialize visible: random pixels + clamped labels
        samples = []
        
        for sample_idx, target_label in enumerate(target_labels):
            # Create label one-hot encoding
            label_onehot = torch.zeros(num_classes, device=device)
            label_onehot[target_label] = 1.0
            
            # Initialize: random pixels + fixed label
            v = torch.cat([
                torch.bernoulli(torch.full((num_pixels,), 0.5, device=device)),
                label_onehot
            ]).unsqueeze(0)  # (1, num_pixels + num_classes)
            
            # Initialize hidden randomly
            h = torch.bernoulli(torch.full((1, model.num_hidden), 0.5, device=device))
            
            # Gibbs sampling with LABEL CLAMPED
            for step in range(burn_in_steps):
                # Update hidden
                _, h = model.mean_field_update(v, h, update_v=False, update_h=True)
                
                # Update pixels (but NOT labels!)
                v_new = v.clone()
                v_new, _ = model.mean_field_update(v_new, h, update_v=True, update_h=False)
                
                # Keep labels clamped, only update pixels
                v[:, :num_pixels] = v_new[:, :num_pixels]
                # v[:, num_pixels:] stays unchanged (clamped label)
                
                if (step + 1) % max(1, burn_in_steps // 5) == 0:
                    print(f"    Sample {sample_idx+1}/{num_samples}, step {step+1}/{burn_in_steps}")
            
            # Extract final pixel values
            final_pixels = v[0, :num_pixels]
            samples.append(final_pixels)
        
        samples = torch.stack(samples, dim=0)
    
    print(f"✓ Generated {num_samples} samples")
    return samples


def sample_conditional_grid(model, label_nodes, num_pixels, grid_shape=(28, 28),
                           samples_per_class=5, burn_in_steps=1000):
    """
    Generate a grid of samples: one row per digit.
    
    Args:
        samples_per_class: How many samples to generate for EACH digit
    
    Returns:
        samples: (num_classes * samples_per_class, num_pixels) tensor
        labels: Corresponding labels for visualization
    """
    num_classes = len(label_nodes)
    
    all_samples = []
    all_labels = []
    
    for digit in range(num_classes):
        target_labels = [digit] * samples_per_class
        digit_samples = sample_conditional(
            model, label_nodes, target_labels, num_pixels, 
            burn_in_steps=burn_in_steps
        )
        all_samples.append(digit_samples)
        all_labels.extend(target_labels)
    
    return torch.cat(all_samples, dim=0), all_labels


def visualize_conditional_samples(samples, labels, grid_shape=(28, 28), 
                                  samples_per_class=5, num_classes=10,
                                  save_path=None):
    """
    Visualize conditional samples in a grid (rows = digits, cols = samples).
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                            figsize=(2*samples_per_class, 2*num_classes))
    
    sample_idx = 0
    for digit in range(num_classes):
        for col in range(samples_per_class):
            ax = axes[digit, col]
            img = samples[sample_idx].cpu().numpy().reshape(grid_shape)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(f'Digit {digit}', fontsize=12, rotation=0, 
                             labelpad=20, va='center')
            sample_idx += 1
    
    fig.suptitle('Conditional Generation: Each Row Shows Samples of One Digit', 
                fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved conditional samples to {save_path}")
    
    plt.show()
    
    return fig