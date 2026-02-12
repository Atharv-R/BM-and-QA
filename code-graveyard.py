"""
Archived Gibbs heuristic/vectorized sampling code.

Purpose:
- Keep the previously removed `gibbs_heur_vectorize` core logic for reference.
- Main codebase now uses coloring-focused sampling and no longer exposes this path.
"""

ARCHIVED_GIBBS_HEUR_VECTORIZED_BRANCH = r'''
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
'''

