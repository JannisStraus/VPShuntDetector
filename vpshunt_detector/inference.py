import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from vpshunt_detector.download import download_weights, resolve_release
from vpshunt_detector.utils import BBox, save_bbox

ALLOWED_FORMAT = {
    ".png",
    ".jpg",
    ".jpeg",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Vote of a single fold for a single image."""

    prediction: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ImageResult:
    """Ensemble outcome for a single image, written as one CSV row."""

    image_name: str
    prediction: str
    confidence: float
    folds: tuple[FoldResult, ...]

    @staticmethod
    def fieldnames(n_folds: int) -> list[str]:
        """Return the CSV columns for an ensemble of `n_folds` folds."""
        per_fold = [
            name
            for i in range(n_folds)
            for name in (f"prediction_fold_{i}", f"confidence_fold_{i}")
        ]
        return ["image_name", *per_fold, "prediction", "confidence"]

    def as_row(self) -> dict[str, str | float]:
        """Flatten into a mapping matching `fieldnames(len(self.folds))`."""
        row: dict[str, str | float] = {"image_name": self.image_name}
        for i, fold in enumerate(self.folds):
            row[f"prediction_fold_{i}"] = fold.prediction
            row[f"confidence_fold_{i}"] = fold.confidence
        row["prediction"] = self.prediction
        row["confidence"] = self.confidence
        return row


@lru_cache(maxsize=1)
def load_models(weights_dir: Path, n_folds: int = 5) -> tuple[YOLO, ...]:
    logger.info("Loading %s fold(s) from '%s'.", n_folds, weights_dir)
    return tuple(
        YOLO(weights_dir / f"fold_{i}" / "best.pt", verbose=True)
        for i in range(n_folds)
    )


def evaluate(
    models: tuple[YOLO, ...], image_path: str | Path, device: str | None = None
) -> tuple[ImageResult, BBox]:
    image_path = Path(image_path)
    confidence_accumulator: dict[str, float] = defaultdict(float)
    bbox_dict: dict[str, tuple[BBox, float]] = {}
    folds: list[FoldResult] = []
    for model in models:
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
        folds.append(FoldResult(pred_cls, pred_conf))
        if pred_cls not in bbox_dict or bbox_dict[pred_cls][1] < pred_conf:
            bbox_dict[pred_cls] = (pred_bbox, pred_conf)

    final_prediction = max(confidence_accumulator, key=confidence_accumulator.get)  # type: ignore
    final_confidence = confidence_accumulator[final_prediction] / len(models)
    result = ImageResult(
        image_name=image_path.name,
        prediction=final_prediction,
        confidence=final_confidence,
        folds=tuple(folds),
    )
    return result, bbox_dict[final_prediction][0]


def infer(
    input_file_or_dir: str | Path,
    output_dir: str | Path,
    instruction_dir: Path | None = None,
    device: str | None = None,
) -> None:
    input_file_or_dir = Path(input_file_or_dir)
    output_dir = Path(output_dir)
    weights_dir = download_weights()
    n_folds = resolve_release().n_folds
    models = load_models(weights_dir, n_folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_instructions: set[str] = set()

    if input_file_or_dir.is_file():
        files = [input_file_or_dir]
    else:
        files = list(input_file_or_dir.iterdir())

    results: list[ImageResult] = []
    for p in tqdm(files, desc="Predicting"):
        image_path = Path(p)
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_FORMAT:
            continue
        result, bbox = evaluate(models, image_path, device)
        results.append(result)

        # BBOX
        save_bbox(
            image_path,
            output_dir / image_path.name,
            bbox,
            result.prediction,
            instruction_dir=instruction_dir,
            missing_instructions=missing_instructions,
        )

    # CSV
    if not results:
        logger.warning(
            "No image with a supported suffix (%s) in '%s'; writing an empty report.",
            ", ".join(sorted(ALLOWED_FORMAT)),
            input_file_or_dir,
        )
    csv_file = output_dir / "results.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ImageResult.fieldnames(len(models)),
        )
        writer.writeheader()
        writer.writerows(result.as_row() for result in results)
