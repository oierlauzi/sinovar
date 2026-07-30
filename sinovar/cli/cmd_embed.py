import argparse
import logging

import numpy as np
import starfile

from .. import embedding

logger = logging.getLogger(__name__)

COMMAND = 'embed'
HELP = 'Compute a manifold embedding from a precomputed distance matrix'


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='STAR',
        help='Input STAR file whose particles will be annotated with the embedding'
    )
    parser.add_argument(
        '-d', '--distance',
        required=True,
        metavar='NPY',
        help='Input squared-distance matrix produced by the `distance` command'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        metavar='STAR',
        help='Output STAR file with embedding data'
    )
    parser.add_argument(
        '--mode',
        choices=sorted(EMBEDDERS),
        default='diffusion',
        help="How to embed the distance matrix: 'diffusion' builds a kNN "
             "affinity graph and takes its diffusion modes (local geometry), "
             "'mds' preserves the distances themselves with landmark "
             "multidimensional scaling (global geometry), 'smacof' refines the "
             "landmark MDS solution by stochastic SMACOF treating the distances "
             "as noisy lower bounds"
    )
    parser.add_argument(
        '--components',
        type=int,
        default=6,
        help='Number of components for dimensionality reduction'
    )
    parser.add_argument(
        '--affinity_k',
        type=int,
        default=4096,
        help='Number of nearest neighbors for the affinity graph '
             '(only used when --mode diffusion)'
    )
    parser.add_argument(
        '--sigma_k',
        type=int,
        default=16,
        help='Neighbor rank used for the adaptive kernel bandwidth '
             '(only used when --mode diffusion)'
    )
    parser.add_argument(
        '--landmarks',
        type=int,
        default=4096,
        help='Number of landmark particles used to anchor the MDS embedding; '
             'values at or above the particle count give exact classical MDS '
             '(only used when --mode mds or --mode smacof, where it sizes the '
             'initial configuration)'
    )
    parser.add_argument(
        '--smacof_steps',
        type=int,
        default=1000,
        help='Number of stochastic SMACOF steps; together with --smacof_batch '
             'this is the I/O budget, as each step reads that many rows of the '
             'distance matrix (only used when --mode smacof)'
    )
    parser.add_argument(
        '--smacof_batch',
        type=int,
        default=256,
        help='Rows of the distance matrix read per SMACOF step; larger batches '
             'give a less noisy step for the same I/O but hold that many matrix '
             'rows on the GPU (only used when --mode smacof)'
    )
    parser.add_argument(
        '--smacof_ridge',
        type=float,
        default=1e-2,
        help='Compactness penalty keeping the one-sided loss from expanding '
             'without bound; larger values shrink the embedding and leave more '
             'lower bounds unmet (only used when --mode smacof)'
    )
    parser.add_argument(
        '--smacof_huber',
        type=float,
        default=0.5,
        help='Violation at which the loss turns linear, in units of the RMS '
             'measured distance; bounds the pull of an over-estimated distance '
             '(only used when --mode smacof)'
    )
    parser.add_argument(
        '--smacof_step_size',
        type=float,
        default=1.0,
        help='Relaxation of the SMACOF step, where 1 minimizes the majorization '
             'exactly; below that damps it (only used when --mode smacof)'
    )
    parser.add_argument(
        '--smacof_momentum',
        type=float,
        default=0.9,
        help='Smoothing of the step direction across batches, averaging '
             'sampling noise over about 1/(1-momentum) batches '
             '(only used when --mode smacof)'
    )


def _compute_diffusion(args: argparse.Namespace, distances2: np.ndarray) -> np.ndarray:
    logger.info('Computing affinity matrix from distances')
    sigma2 = embedding.adaptive_sigma2_median(distances2, k=args.sigma_k)
    affinity = embedding.knn_affinity_from_squared_distance_matrix(
        distances2,
        k=args.affinity_k,
        sigma2=sigma2
    )

    logger.info('Computing the diffusion map embedding')
    return embedding.compute_diffusion_embedding(
        affinity,
        n_components=args.components
    )


def _compute_mds(args: argparse.Namespace, distances2: np.ndarray) -> np.ndarray:
    logger.info('Computing the landmark MDS embedding')
    return embedding.compute_landmark_mds_embedding(
        distances2,
        n_components=args.components,
        n_landmarks=args.landmarks
    )


def _compute_smacof(args: argparse.Namespace, distances2: np.ndarray) -> np.ndarray:
    logger.info('Computing the stochastic SMACOF embedding')
    return embedding.compute_stochastic_smacof_embedding(
        distances2,
        n_components=args.components,
        n_landmarks=args.landmarks,
        n_steps=args.smacof_steps,
        batch_rows=args.smacof_batch,
        ridge=args.smacof_ridge,
        huber=args.smacof_huber,
        step_size=args.smacof_step_size,
        momentum=args.smacof_momentum
    )


#: Embedding strategies selectable with --mode.
EMBEDDERS = {
    'diffusion': _compute_diffusion,
    'mds': _compute_mds,
    'smacof': _compute_smacof,
}


def run(args: argparse.Namespace) -> None:
    logger.info('Reading distance matrix')
    distances2 = np.load(args.distance, mmap_mode='r')

    y = EMBEDDERS[args.mode](args, distances2)

    logger.info('Writing output')
    star = starfile.read(args.input)
    particles_md = star['particles']
    particles_md['sinovarEmbedding'] = [
        '[' + ', '.join(f'{v:.4e}' for v in row) + ']'
        for row in y
    ]
    star['particles'] = particles_md
    starfile.write(star, args.output)
