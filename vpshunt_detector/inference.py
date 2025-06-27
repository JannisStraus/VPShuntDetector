from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from cv2.typing import MatLike
from tqdm import tqdm
from ultralytics import YOLO

from vpshunt_detector.download import download_and_unzip

ALLOWED_FORMAT = {
    ".png",
    ".jpg",
}


def measure_results(
    weights_dir: Path, image: str | Path, device: str | None = None
) -> tuple[dict[str, str | float], tuple[int, int, int, int]]:
    models = [
        YOLO(weights_dir / f"fold_{i}" / "best.pt", verbose=True) for i in range(5)
    ]
    confidence_accumulator: dict[str, float] = defaultdict(float)
    result_dict: dict[str, str | float] = {}
    bbox_dict: dict[str, tuple[tuple[int, int, int, int], float]] = {}
    for i in range(5):
        results = models[i](image, verbose=False, device=device)
        pred_cls = "Nothing"
        pred_conf = 0.0
        pred_bbox = (0, 0, 0, 0)
        for result in results:
            if len(result.boxes.cls) == 0:
                continue

            tmp_conf = float(result.boxes.conf.cpu().numpy()[0])
            if tmp_conf > pred_conf:
                tmp_cls = int(result.boxes.cls.cpu().numpy()[0])
                tmp_bbox = result.boxes.xyxy.cpu().numpy()[0]
                tmp_bbox = [round(i) for i in tmp_bbox]
                pred_cls = models[i].names[tmp_cls]
                pred_conf = tmp_conf
                pred_bbox = tmp_bbox
        confidence_accumulator[pred_cls] += pred_conf
        result_dict[f"prediction_fold_{i}"] = pred_cls
        result_dict[f"confidence_fold_{i}"] = pred_conf
        if pred_cls not in bbox_dict or bbox_dict[pred_cls][1] < pred_conf:
            bbox_dict[pred_cls] = (pred_bbox, pred_conf)

    final_prediction = max(confidence_accumulator, key=confidence_accumulator.get)  # type: ignore
    final_confidence = confidence_accumulator[final_prediction] / len(models)
    result_dict["prediction"] = final_prediction
    result_dict["confidence"] = final_confidence
    return result_dict, bbox_dict[final_prediction][0]


def draw_bbox(
    image_path: str | Path,
    bbox: tuple[int, int, int, int],
    cls: str,
    instruction_dir: str | Path | None = None,
    missing_instructions: set[str] | None = None,
) -> MatLike:
    if missing_instructions is None:
        missing_instructions = set()
    p1, p2 = bbox[:2], bbox[2:]
    img = cv2.imread(str(image_path))
    img = cv2.rectangle(img, p1, p2, color=(0, 255, 0), thickness=4)
    img = cv2.putText(
        img,
        cls,
        (p1[0], max(0, p1[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        thickness=4,
    )
    if not instruction_dir:
        return img

    instruction_dir = Path(instruction_dir)
    manufacturer_path = next(instruction_dir.glob(f"{cls}.*"), None)
    if manufacturer_path:
        manufacturer_img = cv2.imread(str(manufacturer_path))
        manufacturer_img = cv2.resize(
            manufacturer_img,
            (
                int(
                    manufacturer_img.shape[1]
                    * img.shape[0]
                    / manufacturer_img.shape[0]
                ),
                img.shape[0],
            ),
        )
        img = cv2.hconcat([img, manufacturer_img])
    elif cls not in missing_instructions:
        missing_instructions.add(cls)
        print(
            f"Manufacturer image '{cls}.png' or '{cls}.jpg' is missing in '{instruction_dir}'."
        )
    return img


def infer(
    input_file_or_dir: str | Path | list[str | Path],
    output_dir: str | Path,
    instruction_dir: str | Path | None = None,
    device: str | None = None,
) -> None:
    weights_dir = download_and_unzip()
    output_dir = Path(output_dir)
    result_dict: dict[str, list[str | float]] = defaultdict(list)
    missing_instructions: set[str] = set()
    if isinstance(input_file_or_dir, (str, Path)):
        input_file_or_dir = Path(input_file_or_dir)
        if input_file_or_dir.is_file():
            input_file_or_dir = [input_file_or_dir]
        else:
            input_file_or_dir = list(input_file_or_dir.iterdir())

    for p in tqdm(input_file_or_dir, desc="Prediction"):
        image_path = Path(p)
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_FORMAT:
            continue
        result_dict["image_name"].append(image_path.name)
        current_dict, bbox = measure_results(weights_dir, image_path, device)
        for k, v in current_dict.items():
            result_dict[k].append(v)

        # BBOX
        img = draw_bbox(
            image_path,
            bbox,
            str(current_dict["prediction"]),
            instruction_dir=instruction_dir,
            missing_instructions=missing_instructions,
        )
        cv2.imwrite(str(output_dir / image_path.name), img)

    # CSV
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_dir / "results.csv", index=False)
