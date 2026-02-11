import unittest

import numpy as np

from openamundsen_da.util.stats import (
    compute_obs_sigma,
    effective_sample_size,
    normalize_log_weights,
    sigmoid,
    systematic_resample,
)


class StatsTests(unittest.TestCase):
    def test_sigmoid_stability(self):
        x = np.array([-1000.0, 0.0, 1000.0])
        y = sigmoid(x)
        self.assertGreaterEqual(y[0], 0.0)
        self.assertLess(y[0], 1e-6)
        self.assertAlmostEqual(y[1], 0.5, places=12)
        self.assertLessEqual(y[2], 1.0)
        self.assertGreater(y[2], 1.0 - 1e-6)

    def test_normalize_log_weights_and_ess(self):
        lw = np.array([-1000.0, 0.0, -1000.0])
        w = normalize_log_weights(lw)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertGreater(w[1], 0.999)

        w_uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        ess = effective_sample_size(w_uniform)
        self.assertAlmostEqual(ess, 4.0, places=12)

    def test_systematic_resample_output_shape_and_range(self):
        rng = np.random.default_rng(42)
        w = np.array([0.1, 0.2, 0.7], dtype=float)
        idx = systematic_resample(rng, w, n=6)
        self.assertEqual(idx.shape, (6,))
        self.assertTrue(np.all(idx >= 0))
        self.assertTrue(np.all(idx < 3))

    def test_compute_obs_sigma_respects_minimums(self):
        s_fixed = compute_obs_sigma(
            y=0.5,
            n_valid=10,
            cloud_fraction=0.0,
            use_binomial=False,
            sigma_floor=0.05,
            sigma_cloud_scale=0.1,
            min_sigma=0.03,
            obs_sigma=0.2,
        )
        self.assertGreaterEqual(s_fixed, 0.2)

        s_binom = compute_obs_sigma(
            y=0.5,
            n_valid=10,
            cloud_fraction=0.2,
            use_binomial=True,
            sigma_floor=0.05,
            sigma_cloud_scale=0.1,
            min_sigma=0.03,
        )
        self.assertGreaterEqual(s_binom, 0.03)
        self.assertGreater(s_binom, 0.0)


if __name__ == "__main__":
    unittest.main()
