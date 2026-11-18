"""Tests for the PatchCore math on synthetic data (no dataset or weights needed)."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from patchcore import gaussian_blur, greedy_coreset, nearest_distance  # noqa: E402


class CoresetTest(unittest.TestCase):
    def test_size_and_unique(self):
        x = torch.randn(1000, 32)
        idx = greedy_coreset(x, 0.1, proj_dim=16)
        self.assertEqual(len(idx), 100)
        self.assertEqual(len(set(idx.tolist())), 100)

    def test_covers_far_cluster(self):
        # 990 points near 0 and 10 points far away: the coreset must keep a far point.
        x = torch.cat([torch.randn(990, 8) * 0.1, torch.randn(10, 8) * 0.1 + 50])
        idx = greedy_coreset(x, 0.01, proj_dim=8)
        self.assertTrue((idx >= 990).any())


class NearestTest(unittest.TestCase):
    def test_matches_brute_force(self):
        q, b = torch.randn(500, 16), torch.randn(80, 16)
        expected = torch.cdist(q, b).min(dim=1).values
        got = nearest_distance(q, b, chunk=64)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_zero_for_bank_members(self):
        b = torch.randn(50, 4)
        self.assertTrue(torch.allclose(nearest_distance(b, b), torch.zeros(50), atol=1e-3))


class BlurTest(unittest.TestCase):
    def test_keeps_shape_and_mass(self):
        m = torch.zeros(2, 1, 64, 64)
        m[:, :, 32, 32] = 1.0
        out = gaussian_blur(m, sigma=2.0)
        self.assertEqual(out.shape, m.shape)
        self.assertAlmostEqual(float(out[0].sum()), 1.0, places=3)
        self.assertEqual(int(out[0, 0].argmax()), 32 * 64 + 32)


if __name__ == "__main__":
    unittest.main()
