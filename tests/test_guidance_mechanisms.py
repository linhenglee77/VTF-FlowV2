import math

import torch

from TerraFlow.guidance.feasibility_guidance import FlowGuidanceConfig
from TerraFlow.interfaces import SceneBatch
from TerraFlow.models.legacy_transformer_flow import LegacyConditionalTrajectoryFlow
from TerraFlow.planners.legacy_guided_flow_planner import (
    LegacyFlowPlannerConfig,
    LegacyGuidedFlowPlanner,
)
from TerraFlow.terrain.learned_feasibility_field import (
    FeasibilityFieldNet,
    LearnedFieldConfig,
    LearnedTerrainField,
)


def test_vehicle_conditioned_field_uses_speed_and_path_heading():
    torch.manual_seed(11)
    terrain = torch.zeros(1, 3, 32, 32)
    terrain[:, 2] = torch.linspace(0.0, 1.0, 32).view(1, 32, 1)
    model = FeasibilityFieldNet(width=8).eval()
    field = LearnedTerrainField(
        terrain,
        model,
        LearnedFieldConfig(vehicle_physics_enabled=True),
    )
    points = torch.tensor([[[6.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], requires_grad=True)
    slow_forward = field.cost(
        points,
        {"speed": torch.zeros(1, 2), "heading": torch.zeros(1, 2)},
    )
    fast_forward = field.cost(
        points,
        {"speed": torch.full((1, 2), 3.0), "heading": torch.zeros(1, 2)},
    )
    fast_cross = field.cost(
        points,
        {
            "speed": torch.full((1, 2), 3.0),
            "heading": torch.full((1, 2), math.pi / 2),
        },
    )
    assert torch.all(fast_forward >= slow_forward)
    assert not torch.allclose(fast_forward, fast_cross)
    fast_forward.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def test_guidance_history_and_terminal_refinement_are_exposed():
    torch.manual_seed(13)
    model = LegacyConditionalTrajectoryFlow(trajectory_points=8, hidden_dim=64, layers=1)
    scene = SceneBatch(
        ego_history=torch.zeros(1, 1, 3),
        gt_future=torch.zeros(1, 8, 3),
        goal=torch.tensor([[10.0, 1.0, 0.0]]),
        point_cloud=None,
        semantic_labels=None,
        terrain_map=torch.rand(1, 3, 32, 32),
        metadata=[{}],
        vehicle_state={"speed": torch.tensor([1.5]), "heading": torch.tensor([0.0])},
    )
    planner = LegacyGuidedFlowPlanner(
        model,
        residual_std=torch.ones(3),
        metric_scales=torch.tensor([24.0, 12.0, 3.0]),
        config=LegacyFlowPlannerConfig(
            candidates=2,
            integration_steps=3,
            track_feasibility_history=True,
            terminal_refinement_steps=2,
            terminal_refinement_strength=0.005,
            score_unified_objective=True,
        ),
        guidance=FlowGuidanceConfig(enabled=False, strength=0.02),
    )
    prediction = planner(scene)
    assert prediction.diagnostics is not None
    assert prediction.diagnostics["feasibility_cost_history"].shape == (1, 2, 3)
    assert prediction.diagnostics["refinement_cost_history"].shape == (1, 2, 2)
    assert torch.isfinite(prediction.scores).all()
