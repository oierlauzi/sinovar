"""Compare common-line distance estimators under realistic CryoEM noise.

The experiment asks a single question: *how well does a common-line distance
computed from realistic (CTF-modulated, noisy) images recover the ideal
common-line distance computed from clean projections?*

For each trial we

1. pick two volumes (rendered from input PDBs) and a random orientation for
   each,
2. simulate a **clean** projection (no CTF, no noise) and a **realistic** image
   (CTF + additive Gaussian noise at a target SNR) for each orientation,
3. extract the shared common line of the pair from both the clean and the
   realistic images (using the framework's geometry/sinogram primitives),
4. record the *ground truth* = Euclidean distance between the clean common
   lines, and one or more *estimations* = distances computed from the realistic
   common lines.

Repeating this ``N`` times yields a cloud of ``(ground_truth, estimation)``
points whose 2D histogram exposes the estimator's flaws: a vertical offset is
bias, vertical spread is variance, and departure from a monotone trend is
inconsistency.

Note on scale: the framework's CTF-weighted estimator computes
``sum |ctf_j * L_i - ctf_i * L_j|^2``. Even noise-free this equals
``sum (ctf_i ctf_j)^2 |S_i - S_j|^2`` (a CTF^2-reweighting of the clean
distance), and noise adds a positive ``sum (ctf_j^2 var_i + ctf_i^2 var_j)``
floor. The plot is meant to make exactly these effects visible, not to hide
them, so the estimator is reproduced faithfully rather than rescaled.
"""

from typing import Callable, Dict, List, Sequence, Tuple
import argparse
import itertools
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import cryojax.simulator as cxs

from sinovar.geometry import compute_intrinsic_common_line_angles
from sinovar.sinogram import project_sinogram
from sinovar.filter import rfft_multiplicity
from sinovar.ctf import CtfContext, compute_ctf_1d, wiener_ctf_correct_1d

logger = logging.getLogger(__name__)

# A common line only requires the projection along its own angle, so all
# images in a pair are projected with a zero in-plane shift (poses are centred).
_ZERO_SHIFT = jnp.zeros((2,))


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="distance",
        description="Tests common-line distance estimators against a clean ground truth.",
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        nargs="+",
        metavar="PDB",
        help="Input PDB(s). Pairs are drawn across these (including self-pairs) "
             "so the ground-truth distance spans a range; provide several "
             "conformations to populate distances away from zero.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="PNG",
        help="If given, save the figure here (headless) instead of opening a window.",
    )
    parser.add_argument(
        "-n", "--samples",
        type=int,
        default=8192,
        help="Total number of (orientation pair) trials.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Trials simulated per jitted batch (controls peak memory).",
    )
    parser.add_argument(
        "--box_size",
        type=int,
        default=128,
        help="Image side length in pixels.",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=None,
        help="Voxel grid side used to render each PDB (defaults to box_size).",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=1.5,
        help="Pixel/voxel size in angstrom.",
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=300.0,
        help="Acceleration voltage in kV.",
    )
    parser.add_argument(
        "--spherical_aberration",
        type=float,
        default=2.7,
        help="Spherical aberration in mm.",
    )
    parser.add_argument(
        "--amplitude_contrast",
        type=float,
        default=0.1,
        help="Amplitude contrast ratio (q0).",
    )
    parser.add_argument(
        "--defocus_min",
        type=float,
        default=8000.0,
        help="Minimum defocus in angstrom (sampled uniformly per image).",
    )
    parser.add_argument(
        "--defocus_max",
        type=float,
        default=20000.0,
        help="Maximum defocus in angstrom (sampled uniformly per image).",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=0.1,
        help="Signal-to-noise ratio of the realistic images "
             "(noise variance = signal variance / snr).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. 'cpu' or 'gpu:0'. Defaults to the JAX default.",
    )

    return parser.parse_args(argv)


def _select_device(spec: str):
    backend, sep, index = spec.partition(":")
    devices = jax.devices(backend.lower())
    return devices[int(index)] if sep else devices[0]


def load_volume(
    pdb_path: str,
    render_shape: Tuple[int, int, int],
    voxel_size: float,
) -> cxs.FourierVoxelGridVolume:
    """Load a PDB and render it to a Fourier-space voxel grid for fast slicing.

    Rendering every PDB onto the *same* grid shape means all volumes share a
    pytree structure, so a batch of trials for a given volume can be vmapped.
    """
    atom_volume = cxs.load_tabulated_volume(pdb_path)
    return cxs.render_voxel_volume(
        atom_volume,
        cxs.AutoVolumeRenderFn(shape=render_shape, voxel_size=voxel_size),
        output_type=cxs.FourierVoxelGridVolume,
    )


def _make_simulator(
    image_config: cxs.AbstractImageConfig,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    defocus_range: Tuple[float, float],
    snr: float,
) -> Callable:
    """Build a jitted, batched simulator for one volume.

    Returns a function ``(volume, keys) -> (clean, noisy, rotations, defocus)``
    where ``keys`` has shape ``(batch,)``. ``clean`` is the CTF-free, noise-free
    projection (the ideal signal); ``noisy`` adds the CTF and white Gaussian
    noise at the requested SNR.
    """
    defocus_min, defocus_max = defocus_range

    def _simulate_one(volume, key):
        rot_key, defocus_key, noise_key = jax.random.split(key, 3)

        rotation = cxs.EulerAnglePose().rotation.sample_uniform(rot_key)
        pose = cxs.EulerAnglePose.from_rotation(rotation)
        defocus = jax.random.uniform(
            defocus_key, minval=defocus_min, maxval=defocus_max
        )

        clean = cxs.make_image_model(
            volume, image_config, pose
        ).simulate(outputs_real_space=True)

        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(
                defocus_in_angstroms=defocus,
                astigmatism_in_angstroms=0.0,
                spherical_aberration_in_mm=spherical_aberration_mm,
            ),
            amplitude_contrast_ratio=amplitude_contrast,
        )
        signal = cxs.make_image_model(
            volume, image_config, pose, transfer_theory
        ).simulate(outputs_real_space=True)

        #noise_std = jnp.sqrt(jnp.var(signal) / snr)
        noisy = signal + 2.0 * jax.random.normal(noise_key, signal.shape)

        return clean, noisy, rotation.as_matrix(), defocus

    @eqx.filter_jit
    def _simulate_batch(volume, keys):
        return eqx.filter_vmap(_simulate_one, in_axes=(None, 0))(volume, keys)

    return _simulate_batch


# --- common-line extraction and estimators -------------------------------

def _common_line_ft(image: jax.Array, angle: jax.Array) -> jax.Array:
    """Central Fourier slice of ``image`` along ``angle`` (the common line)."""
    line = project_sinogram(image, _ZERO_SHIFT, jnp.atleast_1d(angle))[0]
    return jnp.fft.rfft(line, norm="ortho")

def _abs2(x: jax.Array) -> jax.Array:
    return jnp.square(x.real) + jnp.square(x.imag)

def _weighted_l2(delta: jax.Array, multiplicity: jax.Array) -> jax.Array:
    delta2 = jnp.square(delta.real) + jnp.square(delta.imag)
    return jnp.sum(multiplicity * delta2)


# Estimators map (ft of realistic line i, ft of realistic line j, ctf_i, ctf_j,
# multiplicity) -> squared distance. Add new metrics here to test them.
# Keys are LaTeX math (raw strings; rendered as mathtext in the plot titles).
ESTIMATORS: Dict[str, Callable] = {
    r"|C_j Y_i - C_i Y_j|^2": lambda ft0, ft1, ctf0, ctf1, mult: _weighted_l2(
        ctf1 * ft0 - ctf0 * ft1, mult
    ),
    r"\frac{|C_j Y_i - C_i Y_j|^2}{C_j^2 C_i^2 + \lambda}": lambda ft0, ft1, ctf0, ctf1, mult: jnp.sum(
        mult * _abs2(ctf1 * ft0 - ctf0 * ft1) / (jnp.square(ctf0)*jnp.square(ctf1) + 0.1)
    ),
    r"\frac{|C_j Y_i - C_i Y_j|^2}{(C_j^2 + C_i^2) \sigma + \lambda} - 1": lambda ft0, ft1, ctf0, ctf1, mult: jnp.maximum(
        jnp.sum(
            mult * (_abs2(ctf1 * ft0 - ctf0 * ft1) / ((jnp.square(ctf0) + jnp.square(ctf1))*4.0 + 0.1) - 1)
        ),
        0.0
    ),
    r"|W_i Y_i - W_j Y_j|^2": lambda ft0, ft1, ctf0, ctf1, mult: _weighted_l2(
        wiener_ctf_correct_1d(ft0, ctf0) - wiener_ctf_correct_1d(ft1, ctf1), mult
    ),
}


def _make_pair_metrics(box_size: int, estimator_names: Sequence[str]) -> Callable:
    """Build a vmapped ``(clean/noisy lines, rotations, ctfs) -> (gt, ests)``."""
    multiplicity = rfft_multiplicity(box_size)
    estimators = [(name, ESTIMATORS[name]) for name in estimator_names]

    def _one(clean0, rot0, clean1, rot1, noisy0, noisy1, ctf0, ctf1):
        angle0, angle1 = compute_intrinsic_common_line_angles(rot0, rot1)

        # Ground truth: Euclidean distance between the clean common lines.
        clean_ft0 = _common_line_ft(clean0, angle0)
        clean_ft1 = _common_line_ft(clean1, angle1)
        ground_truth = _weighted_l2(clean_ft0 - clean_ft1, multiplicity)

        # Estimations: distances between the realistic common lines.
        noisy_ft0 = _common_line_ft(noisy0, angle0)
        noisy_ft1 = _common_line_ft(noisy1, angle1)
        estimates = jnp.stack([
            fn(noisy_ft0, noisy_ft1, ctf0, ctf1, multiplicity)
            for _, fn in estimators
        ])
        return ground_truth, estimates

    return eqx.filter_jit(eqx.filter_vmap(_one))


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    estimator_names = list(ESTIMATORS.keys())
    box_size = args.box_size
    render_size = args.render_size or box_size
    render_shape = (render_size, render_size, render_size)

    logger.info("Loading and rendering %d volume(s)", len(args.input))
    volumes = [load_volume(p, render_shape, args.pixel_size) for p in args.input]

    image_config = cxs.BasicImageConfig(
        shape=(box_size, box_size),
        pixel_size=args.pixel_size,
        voltage_in_kilovolts=args.voltage,
        precompute_mode="rfft"
    )
    ctf_context = CtfContext(
        pixel_size_a=args.pixel_size,
        spherical_aberration_mm=args.spherical_aberration,
        voltage_kv=args.voltage,
        q0=args.amplitude_contrast,
    )

    simulate = _make_simulator(
        image_config,
        args.spherical_aberration,
        args.amplitude_contrast,
        (args.defocus_min, args.defocus_max),
        args.snr,
    )
    pair_metrics = _make_pair_metrics(box_size, estimator_names)

    # Distribute the requested trials over every (unordered) pair of volumes,
    # including self-pairs. Self-pairs probe the true-distance ~= 0 regime;
    # cross-pairs (distinct conformations) populate larger distances.
    pairs = list(itertools.combinations_with_replacement(range(len(volumes)), 2))
    if len(volumes) == 1:
        logger.warning(
            "Only one PDB supplied: every ground-truth distance will be ~0 "
            "(common-line consistency floor). Supply several conformations to "
            "span a range of distances."
        )
    per_pair = max(1, -(-args.samples // len(pairs)))  # ceil division

    key = jax.random.key(args.seed)
    gt_all: List[np.ndarray] = []
    est_all: List[np.ndarray] = []
    example_clean = example_noisy = None

    for pair_index, (i, j) in enumerate(pairs):
        logger.info(
            "Pair %d/%d (volumes %d, %d): %d trials",
            pair_index + 1, len(pairs), i, j, per_pair,
        )
        done = 0
        while done < per_pair:
            n = min(args.batch_size, per_pair - done)
            key, key_i, key_j = jax.random.split(key, 3)
            keys_i = jax.random.split(key_i, n)
            keys_j = jax.random.split(key_j, n)

            clean_i, noisy_i, rot_i, defocus_i = simulate(volumes[i], keys_i)
            clean_j, noisy_j, rot_j, defocus_j = simulate(volumes[j], keys_j)

            if example_clean is None:  # keep one (clean, noisy) pair to display
                example_clean = np.asarray(clean_i[0])
                example_noisy = np.asarray(noisy_i[0])

            ctf_i = compute_ctf_1d(defocus_i, box_size, ctf_context)
            ctf_j = compute_ctf_1d(defocus_j, box_size, ctf_context)

            ground_truth, estimates = pair_metrics(
                clean_i, rot_i, clean_j, rot_j,
                noisy_i, noisy_j, ctf_i, ctf_j,
            )
            gt_all.append(np.asarray(ground_truth))
            est_all.append(np.asarray(estimates))
            done += n

    ground_truth = np.concatenate(gt_all)
    estimates = np.concatenate(est_all, axis=0)  # (n_trials, n_estimators)

    _report_and_plot(
        ground_truth, estimates, estimator_names, args,
        example_clean, example_noisy,
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank (Spearman) correlation: monotonic, scale-free consistency measure."""
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _show_example(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _report_and_plot(
    ground_truth: np.ndarray,
    estimates: np.ndarray,
    estimator_names: Sequence[str],
    args: argparse.Namespace,
    example_clean: np.ndarray = None,
    example_noisy: np.ndarray = None,
) -> None:
    n_est = len(estimator_names)
    fig = plt.figure(figsize=(6 * n_est, 8.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.4])

    # Top: one example (clean, noisy) image pair so the noise regime is visible.
    img_axes = top.subplots(1, 2)
    if example_clean is not None:
        _show_example(img_axes[0], example_clean, "clean projection")
        _show_example(img_axes[1], example_noisy,
                      f"realistic (CTF + noise, SNR={args.snr})")

    axes = bottom.subplots(1, n_est, squeeze=False)

    for col, name in enumerate(estimator_names):
        est = estimates[:, col]

        # Absolute agreement is not expected; what matters is a consistent,
        # monotonic trend. Quantify it with Pearson (linear) and Spearman
        # (rank/monotonic) correlation, plus the scatter about the trend line.
        slope, intercept = np.polyfit(ground_truth, est, 1)
        pearson = np.corrcoef(ground_truth, est)[0, 1]
        spearman = _spearman(ground_truth, est)
        scatter = (est - (slope * ground_truth + intercept)).std()

        logger.info(
            "[%s] pearson=%.3f  spearman=%.3f  slope=%.3f  "
            "intercept=%.4e  scatter_about_trend=%.4e",
            name, pearson, spearman, slope, intercept, scatter,
        )

        ax = axes[0, col]
        hb = ax.hexbin(ground_truth, est, gridsize=50, mincnt=1, cmap="viridis")
        bottom.colorbar(hb, ax=ax, label="count")
        xs = np.linspace(ground_truth.min(), ground_truth.max(), 2)
        ax.plot(xs, slope * xs + intercept, "r-", lw=1.5,
                label=f"trend: {slope:.2f}x + {intercept:.1e}")
        ax.set_xlabel("ground truth")
        ax.set_ylabel("estimation")
        ax.set_title(
            f"${name}$\nPearson r = {pearson:.3f}   Spearman ρ = {spearman:.3f}"
        )
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Distance estimators  "
        f"(N={len(ground_truth)}, SNR={args.snr}, box={args.box_size})"
    )
    if args.output is not None:
        fig.savefig(args.output, dpi=150)
        logger.info("Wrote figure to %s", args.output)
    else:
        logger.info("Opening figure window (close it to exit)")
        plt.show()


def main():
    args = _parse_args()
    # Interactive window by default; Agg (headless) only when saving to a file.
    if args.device is not None:
        with jax.default_device(_select_device(args.device)):
            run(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
