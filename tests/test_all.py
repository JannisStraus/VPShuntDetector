import os
import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np
import numpy.testing as npt
import pandas as pd
from dotenv import load_dotenv

from vpshunt_detector.download import (
    download_and_unzip,
    get_cache_dir,
)
from vpshunt_detector.inference import infer
from vpshunt_detector.utils import draw_bbox

load_dotenv()


class Test(unittest.TestCase):
    weights_dir = get_cache_dir() / "VPShuntDetector" / "weights"
    tmp_dir = Path.cwd() / "tmp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir.mkdir()
        shutil.rmtree(cls.weights_dir, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir)

    def test_infer(self) -> None:
        test_data_dir = self.tmp_dir / "examples"
        test_input_dir = test_data_dir / "input"
        test_instructions_dir = test_data_dir / "manufacturer"
        test_output_dir = test_data_dir / "output"
        test_results_file = test_output_dir / "results.csv"
        test_token = os.environ["CLOUD_TEST_TOKEN"]
        test_url = f"https://cloud.uk-essen.de/d/{test_token}/files/"
        test_device = os.getenv("DEVICE", "cpu")
        test_params = {"p": "/examples.zip", "dl": "1"}

        test_data_dir = download_and_unzip(test_data_dir, test_url, test_params)

        # Test inference with single image
        infer(next(test_input_dir.glob("*.jpg")), test_output_dir, device=test_device)
        self.assertTrue(test_results_file.is_file())
        df = pd.read_csv(test_results_file)
        self.assertEqual(len(df), 1)
        shutil.rmtree(test_output_dir)

        # Test inference with single image
        infer(next(test_input_dir.glob("*.jpg")), test_output_dir, device=test_device)
        self.assertTrue(test_results_file.is_file())
        df = pd.read_csv(test_results_file)
        self.assertEqual(len(df), 1)
        shutil.rmtree(test_output_dir)

        # Test inference without instructions
        infer(test_input_dir, test_output_dir, device=test_device)
        self.assertTrue(test_results_file.is_file())
        df = pd.read_csv(test_results_file)
        self.assertEqual(len(df), len(list(test_input_dir.glob("*.jpg"))))
        shutil.rmtree(test_output_dir)

        # Test inference with instructions
        infer(test_input_dir, test_output_dir, test_instructions_dir, test_device)
        self.assertTrue(test_results_file.is_file())
        df = pd.read_csv(test_results_file)
        self.assertEqual(len(df), len(list(test_input_dir.glob("*.jpg"))))
        shutil.rmtree(test_output_dir)

        # Test inference with unexpected missing instructions
        (test_instructions_dir / "Codman Hakim.png").unlink()
        with self.assertLogs("vpshunt_detector.utils", level="WARNING") as cm:
            infer(test_input_dir, test_output_dir, test_instructions_dir, test_device)
        self.assertEqual(len(cm.output), 1)
        self.assertIn(
            (
                "Manufacturer image 'Codman Hakim.png'"
                " or 'Codman Hakim.jpg' is missing in "
            ),
            cm.output[0],
        )
        self.assertTrue(test_results_file.is_file())
        df = pd.read_csv(test_results_file)
        self.assertEqual(len(df), len(list(test_input_dir.glob("*.jpg"))))
        shutil.rmtree(test_output_dir)

    def test_draw_bbox(self) -> None:
        dummy_dir = self.tmp_dir / "dummy"
        dummy_dir.mkdir()
        img_path = dummy_dir / "input.jpg"
        instructions_path = dummy_dir / "test.png"
        ref_size = 750, 250

        expected = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), expected)
        tile = 50
        yy, xx = np.indices(ref_size)
        checker = ((xx // tile + yy // tile) % 2).astype(np.uint8)
        ref_img = np.zeros((*ref_size, 3), dtype=np.uint8)
        ref_img[checker == 0] = (255, 0, 0)  # blue
        ref_img[checker == 1] = (0, 0, 255)  # red
        cv2.imwrite(str(instructions_path), ref_img)

        # Test bbox is None
        img = draw_bbox(img_path, None, "test")
        npt.assert_array_equal(img, expected)

        # Test full
        x1, y1, x2, y2 = 100, 100, 200, 200
        t = 2
        green = np.array([0, 255, 0], dtype=np.uint8)
        img = draw_bbox(img_path, (x1, y1, x2, y2), "test", instruction_dir=dummy_dir)
        # BBox: top and bottom
        npt.assert_array_equal(
            img[y1 : y1 + t, x1 : x2 + 1], np.broadcast_to(green, (t, x2 - x1 + 1, 3))
        )
        npt.assert_array_equal(
            img[y2 - t + 1 : y2 + 1, x1 : x2 + 1],
            np.broadcast_to(green, (t, x2 - x1 + 1, 3)),
        )
        # BBox: left and right
        npt.assert_array_equal(
            img[y1 : y2 + 1, x1 : x1 + t], np.broadcast_to(green, (y2 - y1 + 1, t, 3))
        )
        npt.assert_array_equal(
            img[y1 : y2 + 1, x2 - t + 1 : x2 + 1],
            np.broadcast_to(green, (y2 - y1 + 1, t, 3)),
        )
        # Instructions
        scale = img.shape[0] / ref_img.shape[0]
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        ref_img = cv2.resize(
            ref_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=interpolation,
        )
        npt.assert_array_equal(img[:, expected.shape[1] :], ref_img)


if __name__ == "__main__":
    unittest.main()
