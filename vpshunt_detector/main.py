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
        type=Path,
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
        help="Device for inference (e.g. 'cuda' or 'gpu').",
    )
    args = parser.parse_args()

    infer(
        args.input,
        args.output,
        instruction_dir=args.instructions,
        device=args.device,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
