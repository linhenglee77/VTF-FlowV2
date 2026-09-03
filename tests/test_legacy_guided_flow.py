import torch

from TerraFlow.guidance.feasibility_guidance import FlowGuidanceConfig
from TerraFlow.interfaces import SceneBatch
from TerraFlow.models.legacy_transformer_flow import LegacyConditionalTrajectoryFlow
from TerraFlow.planners.legacy_guided_flow_planner import (
    LegacyFlowPlannerConfig,
    LegacyGuidedFlowPlanner,
)


def test_legacy_flow_checkpoint_architecture_and_guided_sampling():
    torch.manual_seed(3)
    model = LegacyConditionalTrajectoryFlow(trajectory_points=10, hidden_dim=64, layers=1)
    scene = SceneBatch(
        ego_history=torch.zeros(2, 1, 3),
        gt_future=torch.zeros(2, 10, 3),
        goal=torch.tensor([[12.0, 0.0, 0.0], [10.0, 1.0, 0.0]]),
        point_cloud=None,
        semantic_labels=None,
        terrain_map=torch.rand(2, 3, 32, 32),
        metadata=[{}, {}],
    )
    planner = LegacyGuidedFlowPlanner(
        model,
        residual_std=torch.ones(3),
        metric_scales=torch.tensor([24.0, 12.0, 3.0]),
        config=LegacyFlowPlannerConfig(candidates=2, integration_steps=2),
        guidance=FlowGuidanceConfig(enabled=True, strength=0.02),
    )
    prediction = planner(scene)
    assert prediction.trajectories.shape == (2, 2, 10, 3)
    assert prediction.scores.shape == (2, 2)
    assert torch.isfinite(prediction.trajectories).all()
