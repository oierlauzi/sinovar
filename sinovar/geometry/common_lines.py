from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=('norm'))
def find_common_lines(
    directions0: jax.Array,
    directions1: jax.Array,
    norm: bool = False,
) -> jax.Array:
    cross = jnp.linalg.cross(directions0, directions1)

    if norm:
        cross = cross / jnp.linalg.norm(cross, axis=-1, keepdims=True)
    
    return cross

@jax.jit
def compute_intrinsic_angle(
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
) -> Tuple[jax.Array, jax.Array]:
    common_line_direction = find_common_lines(
        matrices0[...,2,:],
        matrices1[...,2,:]
    )
    
    return (
        compute_intrinsic_angle(matrices0, common_line_direction),
        compute_intrinsic_angle(matrices1, common_line_direction)
    )
    