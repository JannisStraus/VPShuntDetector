from collections import defaultdict
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def measure_results(
    weights_dir: Path, image: str | Path, device: str | None = None
) -> dict[str, list[str | float]]:
    models = [
        YOLO(weights_dir / f"fold_{i}" / "best.pt", verbose=True) for i in range(5)
    ]
    confidence_accumulator: dict[str, float] = defaultdict(float)
    result_dict: dict[str, list[str | float]] = defaultdict(list)
    bbox_dict: dict[int, tuple[int, int, int, int]] = {}
    for i in range(5):
        results = models[i](image, verbose=False, device=device)
        pred_cls = "Nothing"
        pred_conf = 0.0
        for result in results:
            if len(result.boxes.cls) == 0:
                continue

            tmp_conf = float(result.boxes.conf.cpu().numpy()[0])
            if tmp_conf > pred_conf:
                tmp_cls = int(result.boxes.cls.cpu().numpy()[0])
                tmp_bbox = result.boxes.xyxy.cpu().numpy()[0]
                pred_cls = models[i].names[tmp_cls]
                pred_conf = tmp_conf
                bbox_dict[i] = tmp_bbox
        confidence_accumulator[pred_cls] += pred_conf
        result_dict[f"prediction_fold_{i}"].append(pred_cls)
        result_dict[f"confidence_fold_{i}"].append(pred_conf)
    final_prediction = max(confidence_accumulator, key=confidence_accumulator.get)  # type: ignore
    final_confidence = confidence_accumulator[final_prediction] / len(models)
    result_dict["prediction"].append(final_prediction)
    result_dict["confidence"].append(final_confidence)
    return result_dict


def infer(
    weights_dir: str | Path,
    input_file_or_dir: str | Path | list[str | Path],
    output_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
    device: str | None = None,
) -> None:
    weights_dir = Path(weights_dir)
    result_dict: dict[str, list[str | float]] = defaultdict(list)
    if isinstance(input_file_or_dir, (str, Path)):
        input_file_or_dir = Path(input_file_or_dir)
        if input_file_or_dir.is_file():
            input_file_or_dir = [input_file_or_dir]
        else:
            input_file_or_dir = list(input_file_or_dir.iterdir())
    for image in input_file_or_dir:
        image_path = Path(image)
        if not image_path.is_file():
            continue
        result_dict["image_name"].append(image_path.name)
        current_dict = measure_results(weights_dir, image, device)
        for k, v in current_dict.items():
            result_dict[k].extend(v)
    if csv_path:
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(csv_path, index=False)
