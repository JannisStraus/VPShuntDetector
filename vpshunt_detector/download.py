import tempfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from vpshunt_detector.utils import get_cache_dir

WEIGHTS_URL = (
    r"https://cloud.uk-essen.de/d/63eb0592c4d94c6bafc9/files/?p=%2Fweights.zip&dl=1"
)


def weights_exist(weights_dir: str | Path) -> bool:
    weights_dir = Path(weights_dir)
    for i in range(5):
        if not (weights_dir / f"fold_{i}" / "best.pt").is_file():
            return False
    return True


def unzip_file(zip_file: str | Path, output_dir: str | Path) -> None:
    zip_file = Path(zip_file)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def download_weights(zip_file: str | Path) -> None:
    response = requests.get(WEIGHTS_URL, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(zip_file, "wb") as file,
        tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Download"
        ) as progress_bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                file.write(chunk)
                progress_bar.update(len(chunk))


def download_and_unzip() -> Path:
    weights_dir = get_cache_dir() / "VPShuntDetector" / "weights"
    if weights_exist(weights_dir):
        return weights_dir

    weights_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_file = Path(temp_dir) / "weights.zip"
        download_weights(zip_file=zip_file)
        unzip_file(zip_file, weights_dir)
    return weights_dir
