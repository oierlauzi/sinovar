import jax
import jax.numpy as jnp

@jax.jit
def _project_image(
    image: jax.Array,
    shift: jax.Array,
    angle: jax.Array
) -> jax.Array:
    h, w = image.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
    yy_c = yy - cy
    xx_c = xx - cx
    
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    # Inverse rotation + inverse shift to find source coordinates
    yy_src = cos_a * yy_c + sin_a * xx_c + cy - shift[0]
    xx_src = -sin_a * yy_c + cos_a * xx_c + cx - shift[1]
    coords = jnp.stack([yy_src.ravel(), xx_src.ravel()])
    rotated = jax.scipy.ndimage.map_coordinates(
        image, 
        coords, 
        order=1, 
        mode='constant', 
        cval=0.0
    )
    return rotated.reshape(h, w).sum(axis=0)


_compute_single_sinogram = jax.jit(
    jax.vmap(_project_image, in_axes=(None, None, 0))
)

compute_sinogram = jax.jit(
    jax.vmap(_compute_single_sinogram, in_axes=(0, 0, None))
)
