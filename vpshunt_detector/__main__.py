import argparse

from vpshunt_detector.inference import infer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect VPShunt valves in X-ray images."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input images.")
    parser.add_argument(
        "-o", "--output", required=True, help="Path to save detection results."
    )
    parser.add_argument(
        "--instructions",
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
        manufacturer_dir=args.instructions,
        device=args.device,
    )


if __name__ == "__main__":
    main()
