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


def download(token: str, zip_file: str | Path) -> None:
    zip_file = Path(zip_file)
    url = f"https://cloud.uk-essen.de/d/{token}/files/"
    params = {"p": f"/{zip_file.name}", "dl": "1"}
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


def download_and_unzip(token: str, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    zip_file = Path(dst_dir) / f"{dst_dir.name}.zip"
    download(token, zip_file)
    unzip(zip_file, dst_dir)
    zip_file.unlink()
    return dst_dir


def download_weights() -> Path:
    weights_dir = get_cache_dir() / "VPShuntDetector" / "weights"
    if not weights_exist(weights_dir):
        download_and_unzip("63eb0592c4d94c6bafc9", weights_dir)
    return weights_dir
