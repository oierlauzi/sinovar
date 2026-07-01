from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp

def _pick_orthogonal_vector(v: jax.Array) -> jax.Array:
    x = v[...,0]
    y = v[...,1]
    z = v[...,2]
    
    zero = jnp.zeros_like(x)
    return jnp.where(
        (jnp.abs(x) > jnp.abs(z))[..., None],
        jnp.stack((-y, x, zero), axis=-1),
        jnp.stack((zero, -z, y), axis=-1)
    )

@partial(jax.jit, static_argnames=('norm'))
def find_common_lines(
    directions0: jax.Array,
    directions1: jax.Array,
    norm: bool = False,
) -> jax.Array:
    cross = jnp.cross(directions0, directions1)
    cross = jnp.where(
        jnp.isclose(jnp.sum(jnp.square(cross), axis=-1, keepdims=True), 0.0),
        _pick_orthogonal_vector(directions0),
        cross
    )

    if norm:
        cross = cross / jnp.linalg.norm(cross, axis=-1, keepdims=True)
    
    return cross

@jax.jit
def compute_dihedral_angles(
    directions0: jax.Array,
    directions1: jax.Array
) -> jax.Array:
    return jnp.acos(jnp.vecdot(directions0, directions1, axis=-1))

@jax.jit
def compute_intrinsic_angles(
    matrices: jax.Array,
    direction: jax.Array
) -> jax.Array:
    vx = matrices[...,0,:]
    vy = matrices[...,1,:]

    px = jnp.vecdot(direction, vx, axis=-1)
    py = jnp.vecdot(direction, vy, axis=-1)
    
    return jnp.atan2(py, px)

@jax.jit
def compute_intrinsic_common_line_angles(
    matrices0: jax.Array,
    matrices1: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    directions0 = matrices0[...,2,:]
    directions1 = matrices1[...,2,:]
    common_line_direction = find_common_lines(directions0, directions1)
    return (
        compute_intrinsic_angles(matrices0, common_line_direction),
        compute_intrinsic_angles(matrices1, common_line_direction),
        compute_dihedral_angles(directions0, directions1)
    )
    