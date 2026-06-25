from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=('n_components', 'alpha', 'diffusion_time'))
def compute_diffusion_embedding(
    affinity: jax.Array,
    n_components: int,
    alpha: float = 1.0,
    diffusion_time: float = 1.0,
) -> jax.Array:
    # Density normalization K_alpha = D^-alpha W D^-alpha decouples the manifold
    # geometry from the sampling density (alpha=1 -> Laplace-Beltrami operator).
    density = affinity.sum(axis=-1)
    d_alpha = density ** (-alpha)
    affinity = affinity * d_alpha[:, None] * d_alpha[None, :]

    # Symmetric conjugate of the row-stochastic diffusion operator P = D^-1 K_alpha.
    # M_sym = D^-1/2 K_alpha D^-1/2 shares P's spectrum but is symmetric.
    degree = affinity.sum(axis=-1)
    d_inv_sqrt = jax.lax.rsqrt(degree)

    normalized = affinity * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

    eigvals, eigvecs = jnp.linalg.eigh(normalized)
    # eigh returns ascending eigenvalues; the diffusion modes are the largest,
    # so reverse and drop the trivial stationary component (lambda = 1).
    eigvals = eigvals[::-1][1:n_components + 1]
    eigvecs = eigvecs[:, ::-1][:, 1:n_components + 1]

    # Right eigenvectors of P recovered as psi = D^-1/2 v, scaled by the
    # diffusion-time weighted eigenvalues lambda^t.
    psi = d_inv_sqrt[:, None] * eigvecs
    embedding = psi * (eigvals ** diffusion_time)[None, :]

    return embedding
