import argparse

from . import cmd_distance
from . import cmd_embed
from .common import configure_logging

_COMMANDS = (cmd_distance, cmd_embed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='sinovar',
        description='Heterogeneity analysis pipeline for CryoEM',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable debug logging'
    )

    subparsers = parser.add_subparsers(dest='command', required=True)
    for module in _COMMANDS:
        subparser = subparsers.add_parser(
            module.COMMAND,
            help=module.HELP,
            description=module.HELP,
        )
        module.add_arguments(subparser)
        subparser.set_defaults(func=module.run)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    args.func(args)
