"""Cognitive PCA: physics latent encoding, decoding, and manifold sampling.

Bridge between KAN coefficient space (W_t) and compact physics latent (z).
Provides the three operations needed by CPPE:
  1. encode: ΔW → z  (not used online — PCA model is frozen)
  2. decode: z → ΔW  (rebuild drift delta from latent)
  3. sample: z_source + α·PC_i with α bounded by observed physics ranges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PCARanges:
    """Per-PC observed range for physics-aware sampling."""
    mins: np.ndarray   # (k,) minimum projection per PC
    maxs: np.ndarray   # (k,) maximum projection per PC

    @classmethod
    def from_projections(cls, z_values: np.ndarray) -> "PCARanges":
        """Compute ranges from observed z projections. z_values: (n_samples, k)."""
        return cls(
            mins=z_values.min(axis=0),
            maxs=z_values.max(axis=0),
        )

    def sample_alpha(self, pc_index: int) -> float:
        """Sample α uniformly within observed range for a given PC."""
        return float(np.random.uniform(self.mins[pc_index], self.maxs[pc_index]))


class CognitivePCA:
    """PCA-based physics latent space for KAN drift coefficients.

    Parameters
    ----------
    pca_path : str
        Path to npz file with keys: mean, Vh, singular, explained, drift_dim.
    k : int, optional
        Number of PCs to use. Defaults to all available. Use 5 for good
        reconstruction (96.8% variance at budget 512).
    pc_ranges : PCARanges, optional
        Observed PC ranges for physics-aware sampling.
    """

    def __init__(
        self,
        pca_path: str,
        k: Optional[int] = None,
        pc_ranges: Optional[PCARanges] = None,
    ):
        data = np.load(pca_path)
        self.mean_: np.ndarray = data["mean"]          # (drift_dim,)
        self.Vh_: np.ndarray = data["Vh"]              # (n_components, drift_dim)
        self.singular_: np.ndarray = data["singular"]  # (n_components,)
        self.explained_: np.ndarray = data["explained"]  # (n_components,)
        self.drift_dim_: int = int(data["drift_dim"])

        if k is not None:
            self.k = min(k, self.Vh_.shape[0])
        else:
            self.k = self.Vh_.shape[0]

        self.pc_ranges = pc_ranges

        # Source physics z: encoding of zero drift delta (ideally near-zero)
        # Computed lazily
        self._z_source: Optional[np.ndarray] = None

    @property
    def z_source(self) -> np.ndarray:
        """Latent representation of source physics (ΔW = 0)."""
        if self._z_source is None:
            self._z_source = self.encode(np.zeros(self.drift_dim_, dtype=np.float32))
        return self._z_source

    # ── encode / decode ──────────────────────────────────────────────

    def encode(self, delta_W: np.ndarray) -> np.ndarray:
        """Project drift delta into PCA latent space.

        Args:
            delta_W: (drift_dim,) or (batch, drift_dim) flattened drift delta.

        Returns:
            z: (k,) or (batch, k) PCA coordinates.
        """
        flat = np.atleast_2d(delta_W.reshape(-1, self.drift_dim_))
        centered = flat - self.mean_[None, :]
        z = centered @ self.Vh_[:self.k, :].T
        if delta_W.ndim == 1:
            return z[0]
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct drift delta from PCA latent.

        Args:
            z: (k,) or (batch, k) PCA coordinates.

        Returns:
            delta_W: (drift_dim,) or (batch, drift_dim) reconstructed drift delta.
        """
        z_flat = np.atleast_2d(z.reshape(-1, self.k))
        reconstructed = self.mean_[None, :] + z_flat @ self.Vh_[:self.k, :]
        if z.ndim == 1:
            return reconstructed[0]
        return reconstructed

    def reconstruction_error(self, delta_W: np.ndarray) -> float:
        """Relative reconstruction error ||ΔW - decode(encode(ΔW))|| / ||ΔW||."""
        recon = self.decode(self.encode(delta_W))
        return float(np.linalg.norm(delta_W - recon) / max(np.linalg.norm(delta_W), 1e-10))

    # ── physics manifold sampling ─────────────────────────────────────

    def sample_z(self, n: int = 1, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample z values along PC directions within observed physics ranges.

        For each sample, randomly picks a PC direction and adds a perturbation
        bounded by the observed range for that PC.

        Args:
            n: number of samples to generate.
            rng: optional random generator for reproducibility.

        Returns:
            z_samples: (n, k) sampled latent vectors.
        """
        if rng is None:
            rng = np.random.default_rng()
        if self.pc_ranges is None:
            raise ValueError(
                "pc_ranges must be set for sampling. "
                "Compute them from observed shift projections first."
            )

        z_src = self.z_source
        samples = np.tile(z_src[None, :], (n, 1)).astype(np.float32)

        for i in range(n):
            pc_idx = int(rng.integers(0, self.k))
            alpha = self.pc_ranges.sample_alpha(pc_idx)
            samples[i, pc_idx] += alpha

        return samples

    def sample_grid(self, pc_dims: tuple[int, ...], n_per_dim: int = 5) -> np.ndarray:
        """Generate a grid of z values sweeping specified PC dimensions.

        Args:
            pc_dims: which PCs to sweep (e.g. (0, 1) for PC1 and PC2).
            n_per_dim: points per dimension.

        Returns:
            z_grid: (n_per_dim^len(pc_dims), k) grid of z values.
        """
        if self.pc_ranges is None:
            raise ValueError("pc_ranges must be set for grid sampling.")

        z_src = self.z_source
        linspaces = [
            np.linspace(self.pc_ranges.mins[d], self.pc_ranges.maxs[d], n_per_dim)
            for d in pc_dims
        ]
        mesh = np.meshgrid(*linspaces, indexing="ij")
        n_total = n_per_dim ** len(pc_dims)

        grid = np.tile(z_src[None, :], (n_total, 1)).astype(np.float32)
        for j, d in enumerate(pc_dims):
            grid[:, d] = mesh[j].ravel()

        return grid

    @property
    def explained_cumulative(self) -> float:
        """Cumulative explained variance ratio for the selected k PCs."""
        return float(np.sum(self.explained_[:self.k]))


def compute_pc_ranges_from_drifts(
    pca: CognitivePCA,
    drift_vectors: dict[str, np.ndarray],
) -> PCARanges:
    """Compute PC ranges by encoding observed drift deltas.

    Args:
        pca: fitted CognitivePCA instance.
        drift_vectors: {name: drift_delta_flat} for observed physics shifts.

    Returns:
        PCARanges with min/max per PC from observed projections.
    """
    all_z = np.stack([
        pca.encode(v) for v in drift_vectors.values()
    ])  # (n_shifts, k)
    return PCARanges.from_projections(all_z)
