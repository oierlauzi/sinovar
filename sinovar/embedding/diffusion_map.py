import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def compute_diffusion_embedding(
    affinity: sp.csr_matrix,
    n_components: int,
    alpha: float = 1.0,
    diffusion_time: float = 1.0,
) -> np.ndarray:
    # Density normalization K_alpha = D^-alpha W D^-alpha decouples the manifold
    # geometry from the sampling density (alpha=1 -> Laplace-Beltrami operator).
    # Conjugating by a diagonal preserves both sparsity and symmetry.
    density = np.asarray(affinity.sum(axis=1)).ravel()
    d_alpha = sp.diags(density ** (-alpha))
    affinity = d_alpha @ affinity @ d_alpha

    # Symmetric conjugate of the row-stochastic diffusion operator P = D^-1 K_alpha.
    # M_sym = D^-1/2 K_alpha D^-1/2 shares P's spectrum but is symmetric.
    degree = np.asarray(affinity.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    d_inv_sqrt_mat = sp.diags(d_inv_sqrt)

    normalized = d_inv_sqrt_mat @ affinity @ d_inv_sqrt_mat

    # The diffusion modes are the largest-algebraic eigenpairs (spectrum <= 1);
    # eigsh on the sparse operator avoids forming a dense matrix.
    eigvals, eigvecs = eigsh(normalized, k=n_components + 1, which='LA')
    # eigsh returns ascending eigenvalues; reverse and drop the trivial
    # stationary component (lambda = 1).
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][1:n_components + 1]
    eigvecs = eigvecs[:, order][:, 1:n_components + 1]

    # Right eigenvectors of P recovered as psi = D^-1/2 v, scaled by the
    # diffusion-time weighted eigenvalues lambda^t.
    psi = d_inv_sqrt[:, None] * eigvecs
    embedding = psi * (eigvals ** diffusion_time)[None, :]

    return embedding
