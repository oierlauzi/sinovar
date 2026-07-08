import argparse
import logging

import numpy as np
import starfile

from ..annotate import (
    CLASS_COLUMN,
    EMBEDDING_COLUMN,
    REDUCERS,
    parse_embedding_column,
)

logger = logging.getLogger(__name__)

COMMAND = 'annotate'
HELP = 'Interactively annotate the 2D embedding and assign a class to each particle'


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='STAR',
        help='Input STAR file annotated with an embedding by the `embed` command'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        metavar='STAR',
        help='Output STAR file with an added `sinovarClassId` column'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=128,
        help='Number of bins per axis for the 2D histogram'
    )
    parser.add_argument(
        '--classes',
        type=int,
        default=3,
        help='Initial number of classes proposed in the GUI'
    )
    parser.add_argument(
        '--covariance-type',
        choices=('full', 'tied', 'diag', 'spherical'),
        default='full',
        help='Initial covariance model for the GMM fits (changeable in the GUI)'
    )
    parser.add_argument(
        '--reduction',
        choices=sorted(REDUCERS),
        default='truncate',
        help="How to reduce the embedding to 2D: 'truncate' keeps the leading "
             "two components, 'pca' projects onto the top principal components, "
             "'umap' runs UMAP (needs `pip install 'sinovar[umap]'`)"
    )
    parser.add_argument(
        '--umap-neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors (only used when --reduction umap)'
    )
    parser.add_argument(
        '--umap-min-dist',
        type=float,
        default=0.1,
        help='UMAP min_dist (only used when --reduction umap)'
    )


def run(args: argparse.Namespace) -> None:
    logger.info('Reading input')
    star = starfile.read(args.input)
    if not isinstance(star, dict) or 'particles' not in star:
        raise ValueError("Input STAR file must contain a 'particles' data block")

    particles_md = star['particles']
    if EMBEDDING_COLUMN not in particles_md.columns:
        raise ValueError(
            f"Column '{EMBEDDING_COLUMN}' not found; run `sinovar embed` first"
        )

    logger.info('Parsing embedding')
    embedding = parse_embedding_column(particles_md[EMBEDDING_COLUMN])
    if not np.all(np.isfinite(embedding)):
        raise ValueError('The embedding contains non-finite values')

    # Import the GUI lazily: matplotlib is an optional dependency, so the rest
    # of the CLI keeps working even when it is not installed.
    try:
        from ..annotate.gui import AnnotationApp
    except ImportError as error:
        raise SystemExit(
            'The annotation GUI requires matplotlib, which is an optional '
            'dependency. Install it with:\n'
            "    pip install 'sinovar[annotate]'"
        ) from error

    # The reduction to the 2D annotation plane is chosen here as the initial
    # projection, but can be changed interactively from the GUI. Building it
    # up front surfaces a missing UMAP dependency before the window opens.
    logger.info('Launching annotation GUI for %d particles', len(embedding))
    try:
        app = AnnotationApp(
            embedding,
            bins=args.bins,
            initial_classes=args.classes,
            covariance_type=args.covariance_type,
            reduction=args.reduction,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
        )
    except ImportError as error:
        raise SystemExit(str(error)) from error

    labels = app.run()

    if labels is None:
        logger.warning('Window closed without saving; output was not written')
        return

    n_classes = int(labels.max()) + 1
    logger.info('Assigned %d particles to %d class(es)', len(labels), n_classes)

    particles_md[CLASS_COLUMN] = labels.astype(int)
    star['particles'] = particles_md

    logger.info('Writing output to %s', args.output)
    starfile.write(star, args.output)
