"""Euler Flow Matching planner with clean-space terrain feasibility guidance."""

from __future__ import annotations

import torch

from TerraFlow.guidance.feasibility_flow_guidance import (
    FeasibilityFlowGuidanceConfig,
    clean_feasibility_gradient,
    scheduled_guidance_strength,
)
from TerraFlow.interfaces import BasePlanner, SceneBatch, TrajectoryBatch
from TerraFlow.guidance.structure_preserving_guidance import (
    adaptive_feasibility_trigger,
    apply_relative_trust_region,
    project_goal_anchored_correction,
)
from TerraFlow.models.flow_network import ConditionalTrajectoryFlow
from TerraFlow.planners.flow_planner import FlowPlanner, FlowPlannerConfig
from TerraFlow.terrain.feasibility_field import TerrainFieldConfig
from TerraFlow.terrain.vehicle_conditioned_field import VehicleConditionedFieldConfig


class GuidedFlowPlanner(BasePlanner):
    """Inject ``-eta(t) grad_x J(x1_hat)`` into the existing Euler dynamics."""

    def __init__(
        self,
        model: ConditionalTrajectoryFlow,
        planner_config: FlowPlannerConfig | None = None,
        guidance_config: FeasibilityFlowGuidanceConfig | None = None,
        terrain_config: TerrainFieldConfig | None = None,
        vehicle_config: VehicleConditionedFieldConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = planner_config or FlowPlannerConfig()
        self.guidance = guidance_config or FeasibilityFlowGuidanceConfig()
        self.terrain_config = terrain_config or TerrainFieldConfig()
        self.vehicle_config = vehicle_config or VehicleConditionedFieldConfig()

    def sample(
        self,
        scene: SceneBatch,
        initial_noise: torch.Tensor | None = None,
    ) -> TrajectoryBatch:
        """Generate paired samples and record per-step feasibility diagnostics."""

        if not self.guidance.enabled or self.guidance.strength == 0.0:
            return FlowPlanner(self.model, self.config).sample(scene, initial_noise)
        scene = scene.as_batch()
        batch = scene.batch_size
        candidates = self.config.candidates
        horizon = self.model.trajectory_points
        expected_shape = (batch, candidates, horizon, 3)
        if initial_noise is None:
            initial_noise = torch.randn(
                expected_shape, dtype=scene.gt_future.dtype,
                device=scene.gt_future.device,
            )
        if initial_noise.shape != expected_shape:
            raise ValueError(f"initial_noise must have shape {expected_shape}")
        if not torch.isfinite(initial_noise).all():
            raise ValueError("initial_noise must contain finite values")
        with torch.no_grad():
            condition = self.model.encode_condition(
                scene.ego_history, scene.goal, scene.terrain_map
            ).repeat_interleave(candidates, dim=0)
        repeated_map = scene.terrain_map.repeat_interleave(candidates, dim=0)
        state = initial_noise.reshape(batch * candidates, horizon, 3).clone()
        history: list[torch.Tensor] = []
        diagnostic_history: dict[str, list[torch.Tensor]] = {}
        step_size = 1.0 / self.config.integration_steps
        for step in range(self.config.integration_steps):
            time = torch.full(
                (batch * candidates,), step * step_size,
                dtype=state.dtype, device=state.device,
            )
            velocity, gradient, diagnostics = clean_feasibility_gradient(
                self.model, state, time, condition, repeated_map, self.guidance,
                self.terrain_config, self.vehicle_config,
            )
            eta = scheduled_guidance_strength(time, self.guidance)
            if self.guidance.adaptive_trigger_enabled:
                trigger = adaptive_feasibility_trigger(
                    diagnostics["guidance_cost"],
                    self.guidance.trigger_alpha,
                    self.guidance.trigger_reference_cost,
                )
            else:
                trigger = torch.ones_like(eta)
            effective_eta = eta * trigger
            proposed_correction = effective_eta[:, None, None] * gradient
            proposed_correction = project_goal_anchored_correction(
                proposed_correction, self.guidance.endpoint_projection
            )
            correction, trust_diagnostics = apply_relative_trust_region(
                proposed_correction,
                velocity,
                self.guidance.trust_region_rho,
                self.guidance.trust_region_scope,  # type: ignore[arg-type]
            )
            state = state + step_size * (velocity - correction)
            if not torch.isfinite(state).all():
                raise RuntimeError(
                    f"guided Euler integration became non-finite at step {step}"
                )
            diagnostics["eta"] = eta.detach()
            diagnostics["trigger"] = trigger.detach()
            diagnostics["effective_eta"] = effective_eta.detach()
            for name, value in trust_diagnostics.items():
                if value.ndim == 2:
                    diagnostics[name] = value.mean(dim=1).detach()
                    diagnostics[f"{name}_max"] = value.max(dim=1).values.detach()
                else:
                    diagnostics[name] = value.detach()
            diagnostics["state_norm"] = torch.linalg.vector_norm(
                state.flatten(start_dim=1), dim=1
            ).detach()
            for name, value in diagnostics.items():
                diagnostic_history.setdefault(name, []).append(value)
            if self.config.save_integration_history:
                history.append(state.detach().clone())
        trajectories = state.reshape(batch, candidates, horizon, 3)
        integration_history = None
        if history:
            integration_history = torch.stack(history, dim=1).reshape(
                batch, candidates, self.config.integration_steps, horizon, 3
            )
        packed: dict[str, torch.Tensor] = {}
        for name, values in diagnostic_history.items():
            stacked = torch.stack(values, dim=1)
            if name == "clean_estimate":
                packed[name] = stacked.reshape(
                    batch, candidates, self.config.integration_steps, horizon, 3
                )
            else:
                packed[name] = stacked.reshape(
                    batch, candidates, self.config.integration_steps
                )
        return TrajectoryBatch(
            trajectories=trajectories,
            scores=None,
            integration_history=integration_history,
            diagnostics=packed,
        )

    def forward(self, scene: SceneBatch) -> TrajectoryBatch:
        """Preserve the common ``prediction = planner(scene)`` interface."""

        return self.sample(scene)


__all__ = ["GuidedFlowPlanner"]
