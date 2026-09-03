"""Dependency-free VTF-Flow unit-test runner."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TerraFlow.tests.test_feasibility_field import (
    test_obstacle_has_lower_feasibility_and_query_is_differentiable,
)
from TerraFlow.tests.test_interfaces import test_scene_and_trajectory_shapes
from TerraFlow.tests.test_legacy_guided_flow import (
    test_legacy_flow_checkpoint_architecture_and_guided_sampling,
)
from TerraFlow.tests.test_learned_feasibility_field import (
    test_learned_field_is_continuous_differentiable_and_speed_conditioned,
)
from TerraFlow.tests.test_receding_horizon import (
    test_receding_horizon_transforms_and_dynamics,
)
from TerraFlow.tests.test_guidance_mechanisms import (
    test_guidance_history_and_terminal_refinement_are_exposed,
    test_vehicle_conditioned_field_uses_speed_and_path_heading,
)


def main() -> None:
    function_tests = [
        test_scene_and_trajectory_shapes,
        test_obstacle_has_lower_feasibility_and_query_is_differentiable,
        test_legacy_flow_checkpoint_architecture_and_guided_sampling,
        test_learned_field_is_continuous_differentiable_and_speed_conditioned,
        test_receding_horizon_transforms_and_dynamics,
        test_vehicle_conditioned_field_uses_speed_and_path_heading,
        test_guidance_history_and_terminal_refinement_are_exposed,
    ]
    for test in function_tests:
        test()
        print(f"PASS {test.__name__}")
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f"{len(function_tests)} function tests and {result.testsRun} unittest cases passed")


if __name__ == "__main__":
    main()
