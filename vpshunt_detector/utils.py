import logging
from pathlib import Path
from typing import Any

import cv2
from cv2.typing import MatLike

BBox = tuple[int, int, int, int] | None
logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def save_bbox(
    image_path: str | Path,
    output_path: str | Path,
    bbox: BBox,
    cls: str,
    **kwargs: Any,
) -> None:
    instruction_dir: str | Path | None = kwargs.get("instruction_dir", None)
    missing_instructions: set[str] | None = kwargs.get("missing_instructions", None)
    img = draw_bbox(image_path, bbox, cls, instruction_dir, missing_instructions)
    cv2.imwrite(str(output_path), img)


def draw_bbox(
    image_path: str | Path,
    bbox: BBox,
    cls: str,
    instruction_dir: str | Path | None = None,
    missing_instructions: set[str] | None = None,
) -> MatLike:
    img = cv2.imread(str(image_path))
    if not bbox:
        return img
    if missing_instructions is None:
        missing_instructions = set()
    p1, p2 = bbox[:2], bbox[2:]
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
                    manufacturer_img.shape[1] * img.shape[0] / manufacturer_img.shape[0]
                ),
                img.shape[0],
            ),
        )
        img = cv2.hconcat([img, manufacturer_img])
    elif cls not in missing_instructions:
        missing_instructions.add(cls)
        logger.warning(
            f"Manufacturer image '{cls}.png' or '{cls}.jpg' is missing in '{instruction_dir}'."
        )
    return img
