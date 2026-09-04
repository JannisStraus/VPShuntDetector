import argparse
import csv
import hashlib
import http.server
import io
import logging
import runpy
import socketserver
import sys
import threading
import unittest
import warnings
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import mock

import cv2
import numpy as np
import numpy.testing as npt
from requests.exceptions import ChunkedEncodingError

from vpshunt_detector.download import (
    ChecksumError,
    download,
    download_and_unzip,
    download_weights,
    load_registry,
    resolve_release,
    unzip,
    weights_exist,
)
from vpshunt_detector.inference import ALLOWED_FORMAT, ImageResult, evaluate, infer
from vpshunt_detector.main import _existing, main
from vpshunt_detector.utils import draw_bbox, get_cache_dir, save_bbox

TESTS_DIR = Path(__file__).parent
CLASSES = (
    "Codman Certas",
    "Codman Hakim",
    "Codman Uni-Shunt",
    "paediGAV",
    "proGAV Gravitationseinheit",
    "proGAV",
    "proSA",
)
TEST_IMAGES = {
    "codmanhakim_2.png": (
        "codman_hakim_programmable/codmanhakim_2.png",
        "90e776317e8a21ce36bf9a03061bb4529640da7347ea3f42343465d89ec05623",
    ),
    "certas_2.png": (
        "codman_certas_plus/certas_2.png",
        "80ec1188fdf5d55f9b4a74404a5f055b6cbdb7d44ca7ab609834ae430251c694",
    ),
}


def write_checkerboard(path: Path, height: int = 250, width: int = 750) -> None:
    tile = 50
    yy, xx = np.indices((height, width))
    checker = ((xx // tile + yy // tile) % 2).astype(np.uint8)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[checker == 0] = (255, 0, 0)
    img[checker == 1] = (0, 0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def write_instructions(directory: Path, classes: tuple[str, ...] = CLASSES) -> None:
    for cls in classes:
        write_checkerboard(directory / f"{cls}.png")


def fetch_test_images() -> Path:
    """Download the pinned fixture radiographs, reusing the cache when warm.

    A cached file is re-hashed rather than trusted, so a corrupted or tampered
    cache heals itself instead of quietly changing what the suite tests.
    """
    dataset_commit = "f0937bf54b07a2ba820a5b1c2cfb7c2a86901576"
    dataset_raw = (
        "https://raw.githubusercontent.com/CSFShuntvalves/xray_csf_shuntvalves"
        f"/{dataset_commit}"
    )
    cache = get_cache_dir() / "test-images" / dataset_commit
    cache.mkdir(parents=True, exist_ok=True)
    for name, (source, sha256) in TEST_IMAGES.items():
        destination = cache / name
        if destination.is_file():
            if hashlib.sha256(destination.read_bytes()).hexdigest() == sha256:
                continue
            destination.unlink()
        download(destination, f"{dataset_raw}/{source}", sha256=sha256)
    return cache


class FakeTensor:
    def __init__(self, array: Any) -> None:
        self._array = np.asarray(array)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> Any:
        return self._array

    def __len__(self) -> int:
        return len(self._array)


class FakeBoxes:
    def __init__(self, cls: Any, conf: Any, xyxy: Any) -> None:
        self.cls = FakeTensor(cls)
        self.conf = FakeTensor(conf)
        self.xyxy = FakeTensor(xyxy)


class FakeResult:
    def __init__(self, cls: Any, conf: Any, xyxy: Any) -> None:
        self.boxes = FakeBoxes(cls, conf, xyxy)


class FakeModel:
    def __init__(self, results: list[FakeResult]) -> None:
        self._results = results
        self.names = dict(enumerate(CLASSES))

    def __call__(self, image_path: Any, **kwargs: Any) -> list[FakeResult]:  # noqa: ARG002
        return self._results


def detection(cls_index: int, conf: float) -> FakeResult:
    return FakeResult([cls_index], [conf], [[10.0, 20.0, 30.0, 40.0]])


def no_detection() -> FakeResult:
    return FakeResult([], [], [])


def run_folds(models: list[FakeModel]) -> tuple[ImageResult, Any]:
    """Run evaluate against stand-in models."""
    return evaluate(cast(Any, tuple(models)), "image.png")


class TestEvaluate(unittest.TestCase):
    """Ensemble aggregation, exercised offline through FakeModel."""

    def test_unanimous(self) -> None:
        result, bbox = run_folds([FakeModel([detection(1, 0.8)]) for _ in range(5)])
        self.assertEqual(result.image_name, "image.png")
        self.assertEqual(result.prediction, "Codman Hakim")
        self.assertAlmostEqual(result.confidence, 0.8, places=5)
        self.assertEqual(bbox, (10, 20, 30, 40))
        self.assertEqual(len(result.folds), 5)
        for fold in result.folds:
            self.assertEqual(fold.prediction, "Codman Hakim")
            self.assertAlmostEqual(fold.confidence, 0.8)

    def test_all_abstain(self) -> None:
        result, bbox = run_folds([FakeModel([no_detection()]) for _ in range(5)])
        self.assertEqual(result.prediction, "Nothing")
        self.assertEqual(result.confidence, 0.0)
        self.assertIsNone(bbox)

    def test_single_fold_decides(self) -> None:
        """Documents current behaviour: abstention carries no weight.

        Four folds see nothing, one fold reports a low-confidence valve, and
        that one fold sets the published prediction.
        """
        models = [FakeModel([no_detection()]) for _ in range(4)]
        models.append(FakeModel([detection(0, 0.30)]))
        result, bbox = run_folds(models)
        self.assertEqual(result.prediction, "Codman Certas")
        self.assertAlmostEqual(result.confidence, 0.06, places=5)
        self.assertIsNotNone(bbox)

    def test_best_bbox_wins(self) -> None:
        _, bbox = run_folds(
            [
                FakeModel([FakeResult([1], [0.4], [[1.0, 2.0, 3.0, 4.0]])]),
                FakeModel([FakeResult([1], [0.9], [[5.0, 6.0, 7.0, 8.0]])]),
            ]
        )
        self.assertEqual(bbox, (5, 6, 7, 8))

    def test_lower_conf_ignored(self) -> None:
        """Covers the branch taken when a later result does not beat the best."""
        result, bbox = run_folds([FakeModel([detection(1, 0.9), detection(0, 0.2)])])
        self.assertEqual(result.prediction, "Codman Hakim")
        self.assertEqual(bbox, (10, 20, 30, 40))

    def test_empty_result_skipped(self) -> None:
        result, _ = run_folds([FakeModel([no_detection(), detection(3, 0.55)])])
        self.assertEqual(result.prediction, "paediGAV")


class TestDrawBBox(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.img_path = self.tmp_dir / "input.png"
        write_checkerboard(self.img_path, 500, 500)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_bbox(self) -> None:
        expected = cv2.imread(str(self.img_path))
        npt.assert_array_equal(draw_bbox(self.img_path, None, "test"), expected)

    def test_box_only(self) -> None:
        img = draw_bbox(self.img_path, (100, 100, 200, 200), "test")
        original = cv2.imread(str(self.img_path))
        self.assertEqual(img.shape, original.shape)
        green = np.array([0, 255, 0], dtype=np.uint8)
        npt.assert_array_equal(
            img[100:102, 100:201], np.broadcast_to(green, (2, 101, 3))
        )
        npt.assert_array_equal(
            img[100:201, 100:102], np.broadcast_to(green, (101, 2, 3))
        )

    def test_instruction_downscale(self) -> None:
        instructions = self.tmp_dir / "instructions"
        write_checkerboard(instructions / "test.png", 1000, 400)
        img = draw_bbox(
            self.img_path, (10, 10, 20, 20), "test", instruction_dir=instructions
        )
        self.assertEqual(img.shape[0], 500)
        self.assertGreater(img.shape[1], 500)

    def test_instruction_upscale(self) -> None:
        instructions = self.tmp_dir / "instructions"
        write_checkerboard(instructions / "test.png", 100, 80)
        img = draw_bbox(
            self.img_path, (10, 10, 20, 20), "test", instruction_dir=instructions
        )
        self.assertEqual(img.shape[0], 500)
        self.assertGreater(img.shape[1], 500)

    def test_missing_warns(self) -> None:
        instructions = self.tmp_dir / "instructions"
        instructions.mkdir()
        seen: set[str] = set()
        with self.assertLogs("vpshunt_detector.utils", level="WARNING") as logs:
            draw_bbox(
                self.img_path,
                (10, 10, 20, 20),
                "Codman Hakim",
                instruction_dir=instructions,
                missing_instructions=seen,
            )
        self.assertEqual(len(logs.output), 1)
        self.assertIn("Codman Hakim", logs.output[0])
        self.assertEqual(seen, {"Codman Hakim"})

    def test_missing_warns_once(self) -> None:
        """A class already in missing_instructions must not warn again."""
        instructions = self.tmp_dir / "instructions"
        instructions.mkdir()
        seen = {"Codman Hakim"}
        logger = "vpshunt_detector.utils"
        with self.assertLogs(logger, level="WARNING") as logs:
            # Emit one unrelated record so assertLogs has something to capture.
            logging.getLogger(logger).warning("sentinel")
            draw_bbox(
                self.img_path,
                (10, 10, 20, 20),
                "Codman Hakim",
                instruction_dir=instructions,
                missing_instructions=seen,
            )
        self.assertEqual(len(logs.output), 1)
        self.assertIn("sentinel", logs.output[0])

    def test_save_bbox(self) -> None:
        output = self.tmp_dir / "out.png"
        save_bbox(self.img_path, output, (10, 10, 20, 20), "test")
        self.assertTrue(output.is_file())
        self.assertIsNotNone(cv2.imread(str(output)))


class TestDownload(unittest.TestCase):
    """Download, checksum and cache behaviour against a local HTTP server."""

    payload: bytes
    good_sha: str
    url: str
    truncate = False
    server: socketserver.TCPServer
    thread: threading.Thread

    @classmethod
    def setUpClass(cls) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for i in range(5):
                archive.writestr(f"weights/fold_{i}/best.pt", f"fold-{i}")
        cls.payload = buffer.getvalue()
        cls.good_sha = hashlib.sha256(cls.payload).hexdigest()

        outer = cls

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(outer.payload)))
                self.end_headers()
                body = outer.payload
                self.wfile.write(body[: len(body) // 2] if outer.truncate else body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        socketserver.TCPServer.allow_reuse_address = True
        cls.server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/weights.zip"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.dst = Path(self._tmp.name) / "release" / "weights"

    def tearDown(self) -> None:
        TestDownload.truncate = False
        self._tmp.cleanup()

    def test_valid_digest(self) -> None:
        """A matching digest extracts, then removes the archive."""
        download_and_unzip(self.dst, self.url, sha256=self.good_sha)
        self.assertTrue(weights_exist(self.dst))
        self.assertFalse(self.dst.with_suffix(".zip").is_file())

    def test_digest_case_insensitive(self) -> None:
        download_and_unzip(self.dst, self.url, sha256=self.good_sha.upper())
        self.assertTrue(weights_exist(self.dst))

    def test_wrong_digest(self) -> None:
        with self.assertRaises(ChecksumError) as ctx:
            download_and_unzip(self.dst, self.url, sha256="0" * 64)
        self.assertIn("expected sha256", str(ctx.exception))
        self.assertFalse(weights_exist(self.dst))
        self.assertFalse(self.dst.with_suffix(".zip").is_file())

    def test_broken_stream(self) -> None:
        """A partial download must leave no archive and nothing extracted."""
        TestDownload.truncate = True
        with self.assertRaises(ChunkedEncodingError):
            download_and_unzip(self.dst, self.url, sha256=self.good_sha)
        self.assertFalse(self.dst.with_suffix(".zip").is_file())

    def test_no_digest(self) -> None:
        download_and_unzip(self.dst, self.url)
        self.assertTrue(weights_exist(self.dst))

    def test_missing_fold(self) -> None:
        """weights_exist is False when any expected fold is absent."""
        download_and_unzip(self.dst, self.url)
        (self.dst / "fold_3" / "best.pt").unlink()
        self.assertFalse(weights_exist(self.dst))
        self.assertTrue(weights_exist(self.dst, n_folds=3))

    def test_unzip_str_paths(self) -> None:
        archive = Path(self._tmp.name) / "plain.zip"
        archive.write_bytes(self.payload)
        unzip(str(archive), str(Path(self._tmp.name) / "out"))
        self.assertTrue(weights_exist(Path(self._tmp.name) / "out" / "weights"))


class TestRegistry(unittest.TestCase):
    def test_default_pinned(self) -> None:
        release = resolve_release()
        self.assertEqual(len(release.sha256), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in release.sha256))
        self.assertTrue(release.url.startswith("https://"))
        self.assertGreaterEqual(release.n_folds, 1)

    def test_default_in_registry(self) -> None:
        default, releases = load_registry()
        self.assertIn(default, releases)
        self.assertIs(resolve_release(default), releases[default])

    def test_unknown_release(self) -> None:
        default, _ = load_registry()
        with self.assertRaises(ValueError) as ctx:
            resolve_release("does-not-exist")
        self.assertIn(default, str(ctx.exception))

    def test_cache_dir_exists(self) -> None:
        self.assertTrue(get_cache_dir().is_dir())

    def test_fetch_once(self) -> None:
        release = resolve_release()

        def fake_download(
            dst: Path,
            url: str,
            params: dict[str, str] | None = None,  # noqa: ARG001
            *,
            sha256: str | None = None,
        ) -> Path:
            # The pinned digest must reach the downloader, not just exist.
            self.assertEqual(url, release.url)
            self.assertEqual(sha256, release.sha256)
            for i in range(release.n_folds):
                (dst / f"fold_{i}").mkdir(parents=True, exist_ok=True)
                (dst / f"fold_{i}" / "best.pt").write_text("stub", encoding="utf-8")
            return dst

        with (
            TemporaryDirectory() as tmp,
            mock.patch(
                "vpshunt_detector.download.get_cache_dir",
                return_value=Path(tmp),
            ),
            mock.patch(
                "vpshunt_detector.download.download_and_unzip",
                side_effect=fake_download,
            ) as downloader,
        ):
            weights_dir = download_weights()
            self.assertTrue(weights_exist(weights_dir, release.n_folds))
            downloader.assert_called_once()

            # A warm cache must not re-download 630 MB.
            self.assertEqual(download_weights(), weights_dir)
            downloader.assert_called_once()

    def test_cache_per_release(self) -> None:
        default, _ = load_registry()
        with (
            TemporaryDirectory() as tmp,
            mock.patch(
                "vpshunt_detector.download.get_cache_dir", return_value=Path(tmp)
            ),
            mock.patch("vpshunt_detector.download.download_and_unzip"),
        ):
            self.assertEqual(
                download_weights().parent.name,
                default,
                "weights must live under a release-specific directory",
            )


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        logger = logging.getLogger("vpshunt_detector")
        handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate

        def restore() -> None:
            logger.handlers[:] = handlers
            logger.setLevel(level)
            logger.propagate = propagate

        self.addCleanup(restore)

    def test_existing_path(self) -> None:
        self.assertEqual(_existing(str(TESTS_DIR)), TESTS_DIR.resolve())

    def test_missing_path(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            _existing(str(TESTS_DIR / "nope"))

    def test_main_args(self) -> None:
        with TemporaryDirectory() as tmp:
            argv = [
                "vpshuntdetector",
                "-i",
                str(TESTS_DIR),
                "-o",
                str(Path(tmp) / "out"),
                "--instructions",
                str(TESTS_DIR),
                "-d",
                "cpu",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("vpshunt_detector.main.infer") as infer_mock,
            ):
                main()
            infer_mock.assert_called_once()
            _, kwargs = infer_mock.call_args
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["instruction_dir"], TESTS_DIR.resolve())

    def test_version_flag(self) -> None:
        with (
            mock.patch.object(sys, "argv", ["vpshuntdetector", "--version"]),
            self.assertRaises(SystemExit) as ctx,
        ):
            main()
        self.assertEqual(ctx.exception.code, 0)

    def test_module_entry(self) -> None:
        """`python -m vpshunt_detector.main` takes the __main__ branch."""
        with TemporaryDirectory() as tmp:
            argv = [
                "vpshuntdetector",
                "-i",
                str(TESTS_DIR),
                "-o",
                str(Path(tmp) / "out"),
                "--verbose",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("vpshunt_detector.inference.infer") as infer_mock,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", RuntimeWarning)
                runpy.run_module("vpshunt_detector.main", run_name="__main__")
        infer_mock.assert_called_once()
        logger = logging.getLogger("vpshunt_detector")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)


class TestInference(unittest.TestCase):
    """End-to-end run over the downloaded CC BY 4.0 radiographs."""

    input_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.input_dir = fetch_test_images()

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.output_dir = self.tmp_dir / "output"
        self.instructions = self.tmp_dir / "instructions"
        write_instructions(self.instructions)
        self.images = sorted(self.input_dir.glob("*.png"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def read_results(self) -> list[dict[str, Any]]:
        results_file = self.output_dir / "results.csv"
        self.assertTrue(results_file.is_file())
        with results_file.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_fixtures(self) -> None:
        self.assertEqual(len(self.images), len(TEST_IMAGES))
        for image in self.images:
            self.assertIn(image.suffix.lower(), ALLOWED_FORMAT)
            self.assertIn(image.name, TEST_IMAGES)

    def test_single_image(self) -> None:
        infer(self.images[0], self.output_dir, device="cpu")
        results = self.read_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["image_name"], self.images[0].name)
        self.assertTrue((self.output_dir / self.images[0].name).is_file())

    def test_directory(self) -> None:
        infer(self.input_dir, self.output_dir, device="cpu")
        results = self.read_results()
        self.assertEqual(len(results), len(self.images))
        for image in self.images:
            self.assertTrue((self.output_dir / image.name).is_file())

    def test_skips_non_images(self) -> None:
        mixed = self.tmp_dir / "mixed"
        mixed.mkdir()
        for image in self.images:
            (mixed / image.name).write_bytes(image.read_bytes())
        (mixed / "notes.txt").write_text("not an image", encoding="utf-8")
        (mixed / "nested").mkdir()
        infer(mixed, self.output_dir, device="cpu")
        self.assertEqual(len(self.read_results()), len(self.images))

    def test_no_images(self) -> None:
        empty = self.tmp_dir / "empty"
        empty.mkdir()
        with self.assertLogs("vpshunt_detector.inference", level="WARNING") as logs:
            infer(empty, self.output_dir, device="cpu")
        self.assertIn(str(empty), logs.output[0])
        # The report is still written, so downstream readers see the columns.
        self.assertEqual(self.read_results(), [])
        with (self.output_dir / "results.csv").open(encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split(",")
        self.assertEqual(header, ImageResult.fieldnames(resolve_release().n_folds))

    def test_with_instructions(self) -> None:
        infer(self.input_dir, self.output_dir, self.instructions, "cpu")
        results = self.read_results()
        self.assertEqual(len(results), len(self.images))
        for image in self.images:
            annotated = cv2.imread(str(self.output_dir / image.name))
            original = cv2.imread(str(image))
            self.assertGreater(annotated.shape[1], original.shape[1])

    def test_missing_instruction(self) -> None:
        infer(self.input_dir, self.output_dir, self.instructions, "cpu")
        predicted = {
            row["prediction"]
            for row in self.read_results()
            if row["prediction"] != "Nothing"
        }
        if not predicted:
            self.skipTest("no valve detected, nothing to look up an instruction for")
        target = sorted(predicted)[0]
        (self.instructions / f"{target}.png").unlink()

        second_output = self.tmp_dir / "output2"
        with self.assertLogs("vpshunt_detector.utils", level="WARNING") as logs:
            infer(self.input_dir, second_output, self.instructions, "cpu")
        self.assertEqual(len(logs.output), 1)
        self.assertIn(target, logs.output[0])


if __name__ == "__main__":
    unittest.main()
