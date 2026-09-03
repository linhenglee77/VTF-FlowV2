"""Unit tests for simple planners using the stable VTF-Flow interfaces."""

from __future__ import annotations

import unittest

import torch

from TerraFlow.interfaces import SceneBatch
from TerraFlow.planners import (
    AStarConfig,
    AStarPlanningError,
    AStarTerrainPlanner,
    ConstantVelocityConfig,
    ConstantVelocityPlanner,
    LocalPathConfig,
    LocalPathPlanner,
    LocalPathUnavailableError,
)


def make_scene(
    *,
    history: torch.Tensor | None = None,
    goal: torch.Tensor | None = None,
    terrain: torch.Tensor | None = None,
    metadata=None,
    horizon: int = 4,
) -> SceneBatch:
    """Build one unbatched synthetic ego-centric scene."""

    return SceneBatch(
        ego_history=(
            history
            if history is not None
            else torch.tensor([[-0.2, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ),
        gt_future=torch.zeros(horizon, 3),
        goal=goal if goal is not None else torch.tensor([3.0, 0.0, 0.0]),
        point_cloud=None,
        semantic_labels=None,
        terrain_map=terrain if terrain is not None else torch.stack(
            (torch.ones(12, 12), torch.zeros(12, 12), torch.full((12, 12), 2.5 / 4.5))
        ),
        metadata={} if metadata is None else metadata,
    )


class ConstantVelocityTests(unittest.TestCase):
    def test_estimates_recent_velocity_and_extrapolates(self) -> None:
        scene = make_scene()
        planner = ConstantVelocityPlanner(
            ConstantVelocityConfig(
                horizon=4, planning_dt_s=0.5, history_dt_s=0.1, velocity_window=2
            )
        )
        prediction = planner(scene)
        self.assertEqual(tuple(prediction.trajectories.shape), (1, 1, 4, 3))
        expected_x = torch.tensor([0.5, 1.0, 1.5, 2.0])
        torch.testing.assert_close(prediction.trajectories[0, 0, :, 0], expected_x)
        torch.testing.assert_close(prediction.trajectories[0, 0, :, 1:], torch.zeros(4, 2))

    def test_single_state_stationary_fallback_is_explicitly_configurable(self) -> None:
        scene = make_scene(history=torch.zeros(1, 3))
        stationary = ConstantVelocityPlanner(
            ConstantVelocityConfig(horizon=3, stationary_fallback=True)
        )(scene)
        torch.testing.assert_close(stationary.trajectories, torch.zeros(1, 1, 3, 3))
        with self.assertRaisesRegex(ValueError, "at least two"):
            ConstantVelocityPlanner(
                ConstantVelocityConfig(horizon=3, stationary_fallback=False)
            )(scene)


class LocalPathTests(unittest.TestCase):
    def test_missing_route_fails_instead_of_inventing_path(self) -> None:
        with self.assertRaisesRegex(LocalPathUnavailableError, "no local route"):
            LocalPathPlanner(LocalPathConfig(horizon=4))(make_scene(metadata={}))

    def test_straight_route_is_smoothly_resampled(self) -> None:
        route = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        prediction = LocalPathPlanner(LocalPathConfig(horizon=4))(
            make_scene(metadata={"local_path": route})
        )
        self.assertEqual(tuple(prediction.trajectories.shape), (1, 1, 4, 3))
        torch.testing.assert_close(
            prediction.trajectories[0, 0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]),
            atol=2e-4,
            rtol=0.0,
        )
        torch.testing.assert_close(prediction.trajectories[0, 0, -1], route[-1])

    def test_batched_metadata_routes(self) -> None:
        one = make_scene(metadata={"route": [[0.0, 0.0], [2.0, 0.0]]})
        two = make_scene(metadata={"route": [[0.0, 0.0], [0.0, 2.0]]})
        batch = SceneBatch(
            ego_history=torch.stack((one.ego_history, two.ego_history)),
            gt_future=torch.stack((one.gt_future, two.gt_future)),
            goal=torch.stack((one.goal, two.goal)),
            point_cloud=None,
            semantic_labels=None,
            terrain_map=torch.stack((one.terrain_map, two.terrain_map)),
            metadata=[one.metadata, two.metadata],
        )
        prediction = LocalPathPlanner(LocalPathConfig(horizon=4))(batch)
        self.assertEqual(tuple(prediction.trajectories.shape), (2, 1, 4, 3))
        self.assertAlmostEqual(float(prediction.trajectories[0, 0, -1, 0]), 2.0)
        self.assertAlmostEqual(float(prediction.trajectories[1, 0, -1, 1]), 2.0)

    def test_shared_metadata_mapping_accepts_batched_route_tensor(self) -> None:
        scene = make_scene()
        batch = SceneBatch(
            ego_history=torch.stack((scene.ego_history, scene.ego_history)),
            gt_future=torch.stack((scene.gt_future, scene.gt_future)),
            goal=torch.stack((scene.goal, scene.goal)),
            point_cloud=None,
            semantic_labels=None,
            terrain_map=torch.stack((scene.terrain_map, scene.terrain_map)),
            metadata={
                "future_route": torch.tensor(
                    [
                        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                    ]
                )
            },
        )
        prediction = LocalPathPlanner(LocalPathConfig(horizon=4))(batch)
        torch.testing.assert_close(
            prediction.trajectories[:, 0, -1],
            torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        )


class AStarTests(unittest.TestCase):
    def config(self, **kwargs) -> AStarConfig:
        return AStarConfig(
            horizon=6,
            forward_extent_m=6.0,
            lateral_extent_m=3.0,
            **kwargs,
        )

    def test_open_grid_reaches_exact_goal(self) -> None:
        scene = make_scene(goal=torch.tensor([5.0, 1.0, 0.2]), horizon=6)
        prediction = AStarTerrainPlanner(self.config())(scene)
        self.assertEqual(tuple(prediction.trajectories.shape), (1, 1, 6, 3))
        torch.testing.assert_close(
            prediction.trajectories[0, 0, -1], scene.goal, atol=1e-6, rtol=0.0
        )
        self.assertEqual(tuple(prediction.scores.shape), (1, 1))

    def test_occupied_cells_are_avoided(self) -> None:
        terrain = make_scene().terrain_map.clone()
        terrain[1, 1:9, 6] = 1.0
        scene = make_scene(goal=torch.tensor([5.0, 0.0, 0.0]), terrain=terrain, horizon=6)
        prediction = AStarTerrainPlanner(self.config())(scene)
        # A barrier along y=0 forces a lateral detour before returning to the goal.
        self.assertGreater(float(prediction.trajectories[0, 0, :, 1].abs().max()), 0.2)
        torch.testing.assert_close(prediction.trajectories[0, 0, -1], scene.goal)

    def test_forbidden_goal_fails_explicitly(self) -> None:
        terrain = make_scene().terrain_map.clone()
        # Goal (5,0) maps to row 10, column 6 for the configured 12x12 grid.
        terrain[1, 10, 6] = 1.0
        scene = make_scene(goal=torch.tensor([5.0, 0.0, 0.0]), terrain=terrain, horizon=6)
        with self.assertRaisesRegex(AStarPlanningError, "goal cell"):
            AStarTerrainPlanner(self.config())(scene)

    def test_out_of_map_goal_fails_instead_of_clipping(self) -> None:
        scene = make_scene(goal=torch.tensor([7.0, 0.0, 0.0]), horizon=6)
        with self.assertRaisesRegex(AStarPlanningError, "outside"):
            AStarTerrainPlanner(self.config())(scene)


if __name__ == "__main__":
    unittest.main()
