import os
import shutil
import unittest
from pathlib import Path

from vpshunt_detector.download import (
    download_and_unzip,
    get_cache_dir,
)
from vpshunt_detector.inference import infer


class Test(unittest.TestCase):
    weights_dir = get_cache_dir() / "VPShuntDetector" / "weights"
    tmp_dir = Path.cwd() / "tmp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir.mkdir(exist_ok=True)
        shutil.rmtree(cls.weights_dir, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir)

    def test_infer(self) -> None:
        test_data_dir = self.tmp_dir / "examples"
        test_input_dir = test_data_dir / "input"
        test_instructions_dir = test_data_dir / "manufacturer"
        test_output_dir = test_data_dir / "output"
        test_token = os.environ["CLOUD_TEST_TOKEN"]
        test_device = os.getenv("DEVICE", "cpu")

        test_data_dir = download_and_unzip(test_token, test_data_dir)
        infer(test_input_dir, test_output_dir, device=test_device)
        infer(test_input_dir, test_output_dir, test_instructions_dir, test_device)


if __name__ == "__main__":
    unittest.main()
