import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from vpshunt_detector.download import download_and_unzip
from vpshunt_detector.utils import BBox, save_bbox

ALLOWED_FORMAT = {
    ".png",
    ".jpg",
}

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_models(weights_dir: Path, n_folds: int = 5) -> tuple[YOLO, ...]:
    logger.info(f"Loading {n_folds} folds from `{weights_dir}`.")
    return tuple(
        YOLO(weights_dir / f"fold_{i}" / "best.pt", verbose=True)
        for i in range(n_folds)
    )


# def _best_detection(result) -> tuple[int, float, BBox] | None:
#     """
#     Return the (class_id, confidence, bbox) of the highest-confidence detection
#     in a single Ultralytics result object, or ``None`` if there is no detection.
#     """
#     boxes = result.boxes
#     if boxes.cls.numel() == 0:  # no detections
#         return None

#     idx = int(boxes.conf.argmax())  # index of max-confidence box
#     conf = float(boxes.conf[idx])
#     cls_id = int(boxes.cls[idx])
#     x1, y1, x2, y2 = boxes.xyxy[idx].round().int().tolist()[:4]
#     return cls_id, conf, (x1, y1, x2, y2)


def measure_results(
    models: tuple[YOLO, ...], image_path: str | Path, device: str | None = None
) -> tuple[dict[str, str | float], BBox]:
    confidence_accumulator: dict[str, float] = defaultdict(float)
    result_dict: dict[str, str | float] = {}
    bbox_dict: dict[str, tuple[BBox, float]] = {}
    for i, model in enumerate(models):
        results = model(image_path, verbose=False, device=device)
        pred_cls = "Nothing"
        pred_conf = 0.0
        pred_bbox: BBox = None
        for result in results:
            if len(result.boxes.cls) == 0:
                continue

            tmp_conf = float(result.boxes.conf.cpu().numpy()[0])
            if tmp_conf > pred_conf:
                tmp_cls = int(result.boxes.cls.cpu().numpy()[0])
                tmp_bbox = result.boxes.xyxy.cpu().numpy()[0]
                tmp_bbox = [round(i) for i in tmp_bbox]
                pred_cls = model.names[tmp_cls]
                pred_conf = tmp_conf
                pred_bbox = tuple(tmp_bbox[:4])
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


# TODO
# def measure_results(
#     models: tuple[YOLO, ...], image_path: str | Path, device: str | None = None
# ) -> tuple[dict[str, str | float], BBox]:
#     confidence_totals: DefaultDict[str, float] = defaultdict(float)
#     result_dict: dict[str, str | float] = {}
#     best_bbox: dict[str, tuple[BBox, float]] = {}
#     for fold_idx, model in enumerate(models):
#         result = model(image_path, verbose=False, device=device)[0]
#         detection = _best_detection(result)
#         if detection is None:
#             cls_name, conf, bbox = "Nothing", 0.0, None
#         else:
#             cls_idx, conf, bbox = detection
#             cls_name = model.names[cls_idx]

#         # accumulate
#         confidence_totals[cls_name] += conf
#         if cls_name not in best_bbox or conf > best_bbox[cls_name][1]:
#             best_bbox[cls_name] = (bbox, conf)

#         result_dict[f"prediction_fold_{fold_idx}"] = cls_name
#         result_dict[f"confidence_fold_{fold_idx}"] = conf
#     final_cls = max(confidence_totals, key=confidence_totals.__getitem__)
#     final_conf = confidence_totals[final_cls] / len(models)

#     result_dict.update(prediction=final_cls, confidence=final_conf)
#     return result_dict, best_bbox[final_cls][0]


def infer(
    input_file_or_dir: Path,
    output_dir: Path,
    instruction_dir: Path | None = None,
    device: str | None = None,
) -> None:
    weights_dir = download_and_unzip()
    models = load_models(weights_dir)
    result_dict: dict[str, list[str | float]] = defaultdict(list)
    missing_instructions: set[str] = set()

    if input_file_or_dir.is_file():
        files = [input_file_or_dir]
    else:
        files = list(input_file_or_dir.iterdir())
    for p in tqdm(files, desc="Predicting"):
        image_path = Path(p)
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_FORMAT:
            continue
        result_dict["image_name"].append(image_path.name)
        current_dict, bbox = measure_results(models, image_path, device)
        for k, v in current_dict.items():
            result_dict[k].append(v)

        # BBOX
        save_bbox(
            image_path,
            output_dir / image_path.name,
            bbox,
            str(current_dict["prediction"]),
            instruction_dir=instruction_dir,
            missing_instructions=missing_instructions,
        )

    # CSV
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_dir / "results.csv", index=False)
