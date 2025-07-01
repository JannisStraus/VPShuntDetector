import shutil
import unittest

from vpshunt_detector.download import download_and_unzip, get_cache_dir, weights_exist


class Test(unittest.TestCase):
    def test_download(self) -> None:
        weights_dir = get_cache_dir() / "VPShuntDetector" / "weights"
        if weights_dir.is_dir():
            shutil.rmtree(weights_dir)
        download_and_unzip()
        self.assertTrue(weights_exist(weights_dir))


if __name__ == "__main__":
    unittest.main()
