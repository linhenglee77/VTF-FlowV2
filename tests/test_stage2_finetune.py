"""Numerical tests for stage-2 TVK fine-tuning helpers."""

from __future__ import annotations

import unittest

from TerraFlow.scripts.fine_tune_vtf_flow import warmup_weight


class WarmupTests(unittest.TestCase):
    def test_linear_warmup_reaches_and_holds_target(self) -> None:
        self.assertAlmostEqual(warmup_weight(0.2, 1, 4), 0.05)
        self.assertAlmostEqual(warmup_weight(0.2, 4, 4), 0.2)
        self.assertAlmostEqual(warmup_weight(0.2, 8, 4), 0.2)

    def test_invalid_values_fail(self) -> None:
        with self.assertRaises(ValueError):
            warmup_weight(-0.1, 1, 2)
        with self.assertRaises(ValueError):
            warmup_weight(0.1, 0, 2)


if __name__ == "__main__":
    unittest.main()
