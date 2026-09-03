"""Unguided Euler sampler for conditional trajectory Flow Matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from TerraFlow.interfaces import BasePlanner, SceneBatch, TrajectoryBatch
from TerraFlow.models.flow_network import ConditionalTrajectoryFlow


SUPPORTED_INTEGRATION_STEPS = (4, 8, 16, 32)


@dataclass(frozen=True)
class FlowPlannerConfig:
    """Sampling count and fixed-step Euler ODE configuration."""

    candidates: int = 8
    integration_steps: int = 16
    save_integration_history: bool = False

    def __post_init__(self) -> None:
        if self.candidates <= 0:
            raise ValueError("candidates must be positive")
        if self.integration_steps not in SUPPORTED_INTEGRATION_STEPS:
            raise ValueError(
                f"integration_steps must be one of {SUPPORTED_INTEGRATION_STEPS}"
            )


class FlowPlanner(BasePlanner):
    """Generate ``K`` trajectories by solving ``dx/dt=v_theta(x,t,c)``.

    Candidates begin from independent ``N(0,I)`` trajectories. No endpoint
    anchoring, terrain scoring, rejection, or feasibility guidance is applied.
    """

    def __init__(
        self,
        model: ConditionalTrajectoryFlow,
        config: FlowPlannerConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config or FlowPlannerConfig()

    @torch.no_grad()
    def sample(
        self,
        scene: SceneBatch,
        initial_noise: torch.Tensor | None = None,
    ) -> TrajectoryBatch:
        """Euler-integrate samples, optionally from fixed noise for exact tests."""

        scene = scene.as_batch()
        batch = scene.batch_size
        candidates = self.config.candidates
        horizon = self.model.trajectory_points
        condition = self.model.encode_condition(
            scene.ego_history, scene.goal, scene.terrain_map
        ).repeat_interleave(candidates, dim=0)
        expected_shape = (batch, candidates, horizon, 3)
        if initial_noise is None:
            initial_noise = torch.randn(
                expected_shape,
                dtype=scene.gt_future.dtype,
                device=scene.gt_future.device,
            )
        if initial_noise.shape != expected_shape:
            raise ValueError(f"initial_noise must have shape {expected_shape}")
        if not torch.isfinite(initial_noise).all():
            raise ValueError("initial_noise must contain finite values")
        state = initial_noise.reshape(batch * candidates, horizon, 3).clone()
        integration = []
        step_size = 1.0 / self.config.integration_steps
        for step in range(self.config.integration_steps):
            time = torch.full(
                (batch * candidates,),
                step * step_size,
                dtype=state.dtype,
                device=state.device,
            )
            state = state + step_size * self.model(state, time, condition)
            if not torch.isfinite(state).all():
                raise RuntimeError("Euler integration produced non-finite trajectories")
            if self.config.save_integration_history:
                integration.append(state.detach().clone())
        trajectories = state.reshape(batch, candidates, horizon, 3)
        history = None
        if integration:
            history = torch.stack(integration, dim=1).reshape(
                batch, candidates, self.config.integration_steps, horizon, 3
            )
        return TrajectoryBatch(
            trajectories=trajectories,
            scores=None,
            integration_history=history,
        )

    def forward(self, scene: SceneBatch) -> TrajectoryBatch:
        """Return unguided Flow samples under the common planner interface."""

        return self.sample(scene)


__all__ = ["FlowPlanner", "FlowPlannerConfig", "SUPPORTED_INTEGRATION_STEPS"]
