"""Wireless physics metrics for the simulator-angle steering-subspace pilot."""

# -----------------------------------------------------------------------------
# PHYSICS PILOT HELPER
# This file isolates the candidate residual from the training loop so it can be
# inspected and tested independently. It measures the fraction of a channel's
# squared-magnitude energy outside the span of its supplied path steering atoms.
# The residual is a research hypothesis, and atoms here are built from simulator
# angles; a low score alone is not evidence that the constraint is scientifically
# appropriate or available from environmental observations at deployment.
# -----------------------------------------------------------------------------

import torch


def steering_subspace_outside_fraction(
    channel: torch.Tensor,
    steering_atoms: torch.Tensor,
    steering_counts: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return per-sample complex channel energy outside its angle-defined subspace.

    Args:
        channel: Real/imaginary channel tensor [B, 2, Nr, Nt].
        steering_atoms: Complex path atoms [B, K, Nr, Nt], padded along K.
        steering_counts: Number of valid atoms for each sample [B].
    """
    if channel.ndim != 4 or channel.shape[1] != 2:
        raise ValueError("Expected channel with shape [B, 2, Nr, Nt].")
    if steering_atoms.ndim != 4 or not torch.is_complex(steering_atoms):
        raise ValueError("Expected complex steering atoms with shape [B, K, Nr, Nt].")
    if steering_atoms.shape[0] != channel.shape[0] or steering_counts.shape[0] != channel.shape[0]:
        raise ValueError("Channel and steering batch dimensions must match.")

    # Convert the model's real/imaginary channels into complex vectors. Flattening
    # makes each [Nr, Nt] channel a single vector for projection onto the atom span.
    h = torch.complex(channel[:, 0], channel[:, 1]).flatten(1)
    scores = []
    for index in range(h.shape[0]):
        count = int(steering_counts[index].item())
        if count < 1:
            raise ValueError("Each channel must have at least one steering atom.")
        # Ignore batch padding, then arrange each path atom as a column. SVD gives
        # an orthonormal basis even when paths are redundant or nearly dependent.
        atoms = steering_atoms[index, :count].flatten(1).transpose(0, 1)
        basis, singular_values, _ = torch.linalg.svd(atoms, full_matrices=False)
        tolerance = (
            max(atoms.shape)
            * torch.finfo(singular_values.dtype).eps
            * singular_values[0]
        )
        rank = int(torch.count_nonzero(singular_values > tolerance).item())
        if rank < 1:
            raise ValueError("Steering atoms have zero numerical rank.")
        q = basis[:, :rank]
        sample = h[index]
        # Orthogonally project the channel onto the supported subspace; the
        # residual is the component that the candidate constraint penalises.
        residual = sample - q @ (q.conj().transpose(0, 1) @ sample)
        outside_energy = torch.sum(torch.abs(residual) ** 2)
        total_energy = torch.sum(torch.abs(sample) ** 2).clamp_min(eps)
        scores.append(outside_energy / total_energy)
    return torch.stack(scores)
# ----------------------------- END PHYSICS PILOT HELPER ----------------------
