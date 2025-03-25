import argparse

from vpshunt_detector.inference import infer


def main() -> None:
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("-i", "--input", required=True, help="TODO")
    parser.add_argument("-o", "--output", required=True, help="TODO")
    parser.add_argument("-r", "--reference-images", required=False, help="TODO")
    parser.add_argument("-d", "--device", required=False, help="TODO")
    args = parser.parse_args()

    infer(
        args.input,
        args.output,
        manufacturer_dir=args.reference_images,
        device=args.device,
    )


if __name__ == "__main__":
    main()
