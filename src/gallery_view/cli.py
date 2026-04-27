"""argparse for ``python -m gallery_view``."""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gallery-view",
        description="Standalone gallery viewer for squid z-stack acquisitions.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="PATH",
        help="Path to an acquisition folder or a parent containing many. "
             "Repeat for multiple sources.",
    )
    return parser.parse_args(argv)
