import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import requests
from tqdm import tqdm

from vpshunt_detector.utils import get_cache_dir

logger = logging.getLogger(__name__)


class ChecksumError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRelease:
    url: str
    sha256: str
    n_folds: int = 5


@lru_cache(maxsize=1)
def load_registry() -> tuple[str, dict[str, ModelRelease]]:
    raw = json.loads(
        files("vpshunt_detector").joinpath("models.json").read_text("utf-8")
    )
    releases = {
        name: ModelRelease(
            url=spec["url"],
            sha256=spec["sha256"],
            n_folds=spec.get("n_folds", 5),
        )
        for name, spec in raw["releases"].items()
    }
    return raw["latest"], releases


def resolve_release(release: str | None = None) -> ModelRelease:
    default, releases = load_registry()
    name = release or default
    try:
        return releases[name]
    except KeyError:
        known = ", ".join(sorted(releases))
        raise ValueError(
            f"Unknown weights release '{name}'. Available releases: {known}."
        ) from None


def weights_exist(weights_dir: str | Path, n_folds: int = 5) -> bool:
    weights_dir = Path(weights_dir)
    return all(
        (weights_dir / f"fold_{i}" / "best.pt").is_file() for i in range(n_folds)
    )


def unzip(zip_file: str | Path, output_dir: str | Path) -> None:
    zip_file = Path(zip_file)
    with zipfile.ZipFile(zip_file, "r", metadata_encoding="utf-8") as zip_ref:
        zip_ref.extractall(output_dir)


def download(
    dst_file: str | Path,
    url: str,
    params: dict[str, str] | None = None,
    *,
    sha256: str | None = None,
) -> None:
    dst_file = Path(dst_file)
    response = requests.get(url, params=params, stream=True, timeout=20.0)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    digest = hashlib.sha256()

    with (
        dst_file.open("wb") as file,
        tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Download"
        ) as progress_bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                file.write(chunk)
                digest.update(chunk)
                progress_bar.update(len(chunk))

    if sha256 is not None and digest.hexdigest() != sha256.lower():
        raise ChecksumError(
            f"Checksum mismatch for '{url}': expected sha256 {sha256.lower()}, "
            f"got {digest.hexdigest()}. The archive was corrupted in transit or "
            f"replaced at the source; refusing to extract it."
        )


def download_and_unzip(
    dst_dir: Path,
    url: str,
    params: dict[str, str] | None = None,
    *,
    sha256: str | None = None,
) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    zip_file = dst_dir.with_suffix(".zip")
    try:
        download(zip_file, url, params, sha256=sha256)
        unzip(zip_file, dst_dir.parent)
    finally:
        zip_file.unlink(missing_ok=True)
    return dst_dir


def download_weights(release: str | None = None) -> Path:
    default, _ = load_registry()
    name = release or default
    model = resolve_release(name)
    # Keyed by release so newly published weights are never shadowed by an
    # older download that happens to have the same fold layout.
    weights_dir = get_cache_dir() / name / "weights"
    if not weights_exist(weights_dir, model.n_folds):
        logger.info("Downloading weights release '%s'.", name)
        download_and_unzip(weights_dir, model.url, sha256=model.sha256)
    return weights_dir
