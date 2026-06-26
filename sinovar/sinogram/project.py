import jax
import jax.numpy as jnp


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

    yy_src = cos_a * yy_c + sin_a * xx_c + cy - shift[1]
    xx_src = -sin_a * yy_c + cos_a * xx_c + cx - shift[0]
    coords = jnp.stack([yy_src.ravel(), xx_src.ravel()])
    rotated = jax.scipy.ndimage.map_coordinates(
        image,
        coords,
        order=1,
        mode='constant',
        cval=0.0
    )
    return rotated.reshape(h, w).sum(axis=0)


project_sinogram = jax.jit(jax.vmap(_project_image, in_axes=(None, None, 0)))
