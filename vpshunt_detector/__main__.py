import argparse
from pathlib import Path

from vpshunt_detector.inference import infer


def _existing(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"'{path}' is not an existing directory")
    return path


def _existing_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"'{path}' is not an existing directory")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect VPShunt valves in X-ray images."
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
        type=_existing_dir,
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

    args.output.mkdir(parents=True, exists_ok=True)
    infer(
        args.input,
        args.output,
        instruction_dir=args.instructions,
        device=args.device,
    )


if __name__ == "__main__":
    main()
