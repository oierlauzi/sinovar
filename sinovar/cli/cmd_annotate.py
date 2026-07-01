import argparse
import logging

import numpy as np
import starfile

from ..annotate import (
    CLASS_COLUMN,
    EMBEDDING_COLUMN,
    TruncationReducer,
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
        default=182,
        help='Number of bins per axis for the 2D histogram'
    )
    parser.add_argument(
        '--classes',
        type=int,
        default=3,
        help='Initial number of classes proposed in the GUI'
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

    # Reduce to the 2D annotation plane. Truncation keeps the leading
    # (most significant) diffusion-map components; swap in another Reducer
    # here to support PCA/UMAP/... in the future.
    points = TruncationReducer(n_components=2).reduce(embedding)

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

    logger.info('Launching annotation GUI for %d particles', len(points))
    app = AnnotationApp(points, bins=args.bins, initial_classes=args.classes)
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
