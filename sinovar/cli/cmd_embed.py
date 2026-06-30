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
        '--components',
        type=int,
        default=6,
        help='Number of components for dimensionality reduction'
    )
    parser.add_argument(
        '--affinity_k',
        type=int,
        default=4096,
        help='Number of nearest neighbors for the affinity graph'
    )
    parser.add_argument(
        '--sigma_k',
        type=int,
        default=16,
        help='Neighbor rank used for the adaptive kernel bandwidth'
    )


def run(args: argparse.Namespace) -> None:
    logger.info('Reading distance matrix')
    distances2 = np.load(args.distance, mmap_mode='r')

    logger.info('Computing affinity matrix from distances')
    sigma2 = embedding.adaptive_sigma2_median(distances2, k=args.sigma_k)
    affinity = embedding.knn_affinity_from_squared_distance_matrix(
        distances2,
        k=args.affinity_k,
        sigma2=sigma2
    )

    logger.info('Computing the embedding')
    y = embedding.compute_diffusion_embedding(
        affinity,
        n_components=args.components
    )

    logger.info('Writing output')
    star = starfile.read(args.input)
    particles_md = star['particles']
    particles_md['sinovarEmbedding'] = [
        '[' + ', '.join(f'{v:.4e}' for v in row) + ']'
        for row in y
    ]
    star['particles'] = particles_md
    starfile.write(star, args.output)
