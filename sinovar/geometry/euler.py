import jax
import jax.numpy as jnp

def euler_zyz_to_matrix(
    rot: jax.Array,
    tilt: jax.Array,
    psi: jax.Array
) -> jax.Array:
    
    # Create the output
    batch_shape = jnp.broadcast_shapes(rot.shape, tilt.shape, psi.shape)
    result_shape = batch_shape + (3, 3)

    ai = rot
    aj = tilt
    ak = psi 

    # Obtain sin and cos of the half angles
    ci = jnp.cos(ai)
    si = jnp.sin(ai)
    cj = jnp.cos(aj)
    sj = jnp.sin(aj)
    ck = jnp.cos(ak)
    sk = jnp.sin(ak)
    
    # Obtain the combinations
    cc = cj * ci
    cs = cj * si
    sc = sj * ci
    ss = sj * si

    # Build the matrix
    result = jnp.stack(
        [
            ck*cc - sk*si,  
            ck*cs + sk*ci,  
            -ck*sj,
            -sk*cc - ck*si, 
            -sk*cs + ck*ci,  
            sk*sj,
            sc,              
            ss,              
            cj,
        ], 
        axis=-1
    )
    result = result.reshape(result_shape)

    return result
