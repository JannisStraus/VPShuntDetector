import argparse
import logging
from importlib.metadata import version
from pathlib import Path

from vpshunt_detector.inference import infer


def _existing(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"'{path}' is not an existing directory")
    return path


def _target(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    package_logger = logging.getLogger("vpshunt_detector")
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO if verbose else logging.WARNING)
    package_logger.propagate = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect VPShunt valves in X-ray images."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {version('vpshunt-detector')}",
    )
    parser.add_argument(
        "-i", "--input", type=_existing, required=True, help="Path to input images."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=_target,
        required=True,
        help="Path to save detection results.",
    )
    parser.add_argument(
        "--instructions",
        type=_existing,
        required=False,
        help="Directory with instruction images for valve models.",
    )
    parser.add_argument(
        "-d",
        "--device",
        required=False,
        help="Device for inference (e.g. 'cuda' or 'cpu'). Default: auto-detect.",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        default=False,
        help="Log detailed informations.",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    infer(
        args.input,
        args.output,
        instruction_dir=args.instructions,
        device=args.device,
    )


if __name__ == "__main__":
    main()
