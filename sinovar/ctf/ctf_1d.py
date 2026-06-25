from functools import partial
import jax
import jax.numpy as jnp

from .ctf_context import CtfContext

@partial(jax.jit, static_argnames=('box_size', "context"))
def compute_ctf_1d(
    defocus_a: jnp.ndarray,
    box_size: int,
    context: CtfContext
):
    k = jnp.fft.rfftfreq(box_size, d=context.pixel_size_a)
    k2 = jnp.square(k)
    
    wavelength = context.wavelength_a
    wavelength2 = wavelength*wavelength
    spherical_aberration = context.spherical_aberration_a
    q0 = context.q0
    
    angle = jnp.pi*wavelength*k2*(0.5*spherical_aberration*wavelength2*k2 - defocus_a[...,None])
    return -jnp.sin(angle - context.amplitude_contrast_phase_shift)
