"""Check trained-kernel inference and edge cases without TensorFlow."""
import unittest

import numpy as np

from blur_detection import FourierNetwork, block_scores, model_path, operation_counts


class InferenceTests(unittest.TestCase):
    def test_untrimmed_model_matches_real_fft(self):
        model = FourierNetwork(model_path(128, 128))
        x = np.random.default_rng(7).standard_normal((16, 128)).astype("float32")
        mse = np.mean((model.transform(x) - np.fft.fft(x).real) ** 2)
        self.assertLess(mse, 0.01)

    def test_batched_two_dimensional_transform_matches_row_column_calls(self):
        model = FourierNetwork(model_path(128, 72))
        x = np.random.default_rng(4).standard_normal((128, 128)).astype("float32")
        rows = np.array([model.transform(row) for row in x])
        expected = np.array([model.transform(col) for col in rows.T]).T
        np.testing.assert_allclose(model.transform2d(x), expected, atol=0.002, rtol=0.002)

    def test_constant_image_scores_are_finite(self):
        scores = block_scores(np.full((256, 384), 128), 128)
        self.assertEqual(scores.shape, (2, 3))
        self.assertTrue(np.isfinite(scores).all())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            block_scores(np.zeros((127, 127)), 128)
        with self.assertRaises(ValueError):
            block_scores(np.zeros((128, 128)), 128, mask_radius=64)
        with self.assertRaises(ValueError):
            FourierNetwork(model_path(128, 72)).transform(np.zeros(64))

    def test_paper_configuration_reduces_operations_by_about_44_percent(self):
        trimmed = sum(operation_counts(128, 72).values())
        full = sum(operation_counts(128, 128).values())
        self.assertAlmostEqual(1 - trimmed / full, 0.4383578431372549)


if __name__ == "__main__":
    unittest.main()
