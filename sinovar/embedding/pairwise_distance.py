import jax
import jax.numpy as jnp

@jax.jit
def compute_pairwise_distance2(
    images: jax.Array,
    ctfs: jax.Array,
    multiplicity: jax.Array
) -> jax.Array:
    # Expand: 
    # |A_i*c_j - A_j*c_i|^2
    # |A_i|^2*c_j^2 + |A_j|^2*c_i^2 - 2*c_i*c_j*Re(A_i*conj(A_j))
    # term2 == term1.T by symmetry, so only 3 matmuls needed instead of 4.
    n = images.shape[0]
    A = images.reshape(n, -1)
    c = ctfs.reshape(n, -1)
    w = multiplicity.reshape(-1)

    A_abs2 = jnp.square(A.real) + jnp.square(A.imag)
    cr = c * A.real
    ci = c * A.imag

    term1 = (A_abs2 * w) @ (c**2).T
    term2 = term1.T
    term3 = (cr * w) @ cr.T + (ci * w) @ ci.T

    return jnp.maximum(term1 + term2 - 2*term3, 0.0)
