import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from vpshunt_detector.utils import get_cache_dir


def weights_exist(weights_dir: str | Path) -> bool:
    weights_dir = Path(weights_dir)
    return all((weights_dir / f"fold_{i}" / "best.pt").is_file() for i in range(5))


def unzip(zip_file: str | Path, output_dir: str | Path) -> None:
    zip_file = Path(zip_file)
    with zipfile.ZipFile(zip_file, "r", metadata_encoding="utf-8") as zip_ref:
        zip_ref.extractall(output_dir)


def download(
    zip_file: str | Path, url: str, params: dict[str, str] | None = None
) -> None:
    zip_file = Path(zip_file)
    response = requests.get(url, params=params, stream=True, timeout=20.0)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        zip_file.open("wb") as file,
        tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Download"
        ) as progress_bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                file.write(chunk)
                progress_bar.update(len(chunk))


def download_and_unzip(
    dst_dir: Path, url: str, params: dict[str, str] | None = None
) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    zip_file = dst_dir.with_suffix(".zip")
    download(zip_file, url, params)
    unzip(zip_file, dst_dir.parent)
    zip_file.unlink()
    return dst_dir


def download_weights() -> Path:
    weights_dir = get_cache_dir() / "VPShuntDetector" / "weights"
    weights_url = (
        "https://github.com/JannisStraus/VPShuntDetector"
        "/releases/download/v0.1.3/weights.zip"
    )
    if not weights_exist(weights_dir):
        download_and_unzip(weights_dir, weights_url)
    return weights_dir
