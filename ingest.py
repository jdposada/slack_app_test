import argparse
import logging
import os
import sys

from omop_index import build_default_index


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("omop54_ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the packaged OMOP 5.4 SQLite index.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("OMOP_INDEX_PATH", "data/omop54.db"),
        metavar="PATH",
        help="Where to write the SQLite index file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_default_index(args.output)
    logger.info("OMOP 5.4 index written to %s", args.output)


if __name__ == "__main__":
    main()
