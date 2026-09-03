"""Tests for configuration-driven hyperparameters."""

import json
import tempfile
import unittest
from pathlib import Path

from TerraFlow.configs import ExperimentConfig, load_config


class ConfigurationTest(unittest.TestCase):
    """Verify configuration loading and validation."""

    def test_default_config_has_no_hard_coded_dataset_path(self) -> None:
        """A fresh checkout is independent of any machine's dataset layout."""

        config_path = Path(__file__).parents[1] / "configs" / "default.json"
        config = load_config(config_path)
        self.assertIsInstance(config, ExperimentConfig)
        self.assertIsNone(config.dataset.root)
        self.assertEqual(config.dataset.future_steps, config.planner.horizon)
        self.assertEqual(config.planner.trajectory_dim, 3)

    def test_config_rejects_inconsistent_horizons(self) -> None:
        """Dataset targets and planner outputs must agree on H."""

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {"future_steps": 10},
                        "planner": {"horizon": 20},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "future_steps"):
                load_config(config_path)

    def test_config_rejects_unknown_top_level_keys(self) -> None:
        """Misspelled configuration sections do not fail silently."""

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text('{"dataaset": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown top-level"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
