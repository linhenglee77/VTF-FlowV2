"""Checkpoint-stable guided planner for the five Transformer Flow runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from TerraFlow.guidance.feasibility_guidance import (
    FlowGuidanceConfig,
    feasibility_gradient,
    trajectory_vehicle_state,
)
from TerraFlow.interfaces import BasePlanner, SceneBatch, TrajectoryBatch
from TerraFlow.terrain.feasibility_field import AnalyticTerrainField, TerrainFieldConfig
from TerraFlow.terrain.learned_feasibility_field import (
    FeasibilityFieldNet,
    LearnedFieldConfig,
    LearnedTerrainField,
)


@dataclass(frozen=True)
class LegacyFlowPlannerConfig:
    candidates: int = 8
    integration_steps: int = 16
    anchor_endpoint: bool = True
    save_integration_history: bool = False
    track_feasibility_history: bool = False
    terminal_refinement_steps: int = 0
    terminal_refinement_strength: float = 0.04
    score_unified_objective: bool = False


class LegacyGuidedFlowPlanner(BasePlanner):
    """Euler solve with optional analytic or learned feasibility gradients."""

    def __init__(
        self,
        model,
        residual_std: torch.Tensor,
        metric_scales: torch.Tensor,
        config: LegacyFlowPlannerConfig | None = None,
        guidance: FlowGuidanceConfig | None = None,
        terrain_config: TerrainFieldConfig | None = None,
        terrain_field_model: FeasibilityFieldNet | None = None,
        learned_field_config: LearnedFieldConfig | None = None,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("residual_std", residual_std.float().reshape(1, 1, 3))
        self.register_buffer("metric_scales", metric_scales.float().reshape(1, 1, 3))
        self.config = config or LegacyFlowPlannerConfig()
        self.guidance = guidance or FlowGuidanceConfig(enabled=False)
        self.terrain_config = terrain_config or TerrainFieldConfig()
        self.terrain_field_model = terrain_field_model
        self.learned_field_config = learned_field_config or LearnedFieldConfig()
        if self.terrain_field_model is not None:
            self.terrain_field_model.eval()
            for parameter in self.terrain_field_model.parameters():
                parameter.requires_grad_(False)

    def _metric_path(self, state, goal_normalized):
        alpha = torch.linspace(
            1.0 / state.shape[1], 1.0, state.shape[1],
            device=state.device, dtype=state.dtype,
        ).view(1, -1, 1)
        normalized = state * self.residual_std + alpha * goal_normalized[:, None, :]
        return normalized * self.metric_scales

    def forward(self, scene: SceneBatch) -> TrajectoryBatch:
        scene = scene.as_batch()
        terrain = scene.terrain_map
        goal_normalized = scene.goal / self.metric_scales.view(1, 3)
        batch, candidates = len(terrain), self.config.candidates
        condition = self.model.encode_condition(terrain, goal_normalized)
        condition = condition.repeat_interleave(candidates, dim=0)
        repeated_goal = goal_normalized.repeat_interleave(candidates, dim=0)
        repeated_terrain = terrain.repeat_interleave(candidates, dim=0)
        repeated_vehicle_state = None
        if scene.vehicle_state is not None:
            repeated_vehicle_state = {
                key: value.to(terrain).reshape(batch).repeat_interleave(candidates)
                for key, value in scene.vehicle_state.items()
            }
        if self.terrain_field_model is None:
            field = AnalyticTerrainField(repeated_terrain, self.terrain_config)
        else:
            field = LearnedTerrainField(
                terrain, self.terrain_field_model, self.learned_field_config
            ).repeat_interleave(candidates)
        state = torch.randn(
            batch * candidates, self.model.trajectory_points, 3,
            device=terrain.device, dtype=terrain.dtype,
        )
        history = []
        feasibility_history = []
        step_size = 1.0 / self.config.integration_steps
        for step in range(self.config.integration_steps):
            time_value = step / self.config.integration_steps
            time = torch.full((len(state),), time_value, device=state.device, dtype=state.dtype)
            with torch.no_grad():
                velocity = self.model(state, time, condition=condition)
            eta = self.guidance.eta(time_value)
            if eta > 0.0 or self.config.track_feasibility_history:
                gradient, cost_terms = feasibility_gradient(
                    state,
                    lambda value: self._metric_path(value, repeated_goal),
                    repeated_terrain,
                    self.guidance,
                    self.terrain_config,
                    field=field,
                    initial_vehicle_state=repeated_vehicle_state,
                )
                if eta > 0.0:
                    velocity = velocity - eta * gradient
                if self.config.track_feasibility_history:
                    feasibility_history.append(cost_terms["total_cost"])
            state = (state + step_size * velocity).clamp(-5.0, 5.0)
            if self.config.anchor_endpoint:
                state[:, -1] = 0.0
            if self.config.save_integration_history:
                history.append(self._metric_path(state, repeated_goal).detach())
        refinement_history = []
        for _ in range(self.config.terminal_refinement_steps):
            gradient, cost_terms = feasibility_gradient(
                state,
                lambda value: self._metric_path(value, repeated_goal),
                repeated_terrain,
                self.guidance,
                self.terrain_config,
                field=field,
                initial_vehicle_state=repeated_vehicle_state,
            )
            refinement_history.append(cost_terms["total_cost"])
            state = (
                state - self.config.terminal_refinement_strength * gradient
            ).clamp(-5.0, 5.0)
            if self.config.anchor_endpoint:
                state[:, -1] = 0.0
        metric = self._metric_path(state, repeated_goal)
        vehicle_state = (
            trajectory_vehicle_state(
                metric, self.guidance.planning_dt_s, repeated_vehicle_state
            )
            if self.guidance.vehicle_conditioned
            else None
        )
        if self.config.score_unified_objective:
            _, score_terms = feasibility_gradient(
                state,
                lambda value: self._metric_path(value, repeated_goal),
                repeated_terrain,
                self.guidance,
                self.terrain_config,
                field=field,
                initial_vehicle_state=repeated_vehicle_state,
            )
            scores = score_terms["total_cost"].reshape(batch, candidates)
        else:
            scores = field.cost(metric, vehicle_state).mean(dim=1).reshape(batch, candidates)
        trajectories = metric.reshape(batch, candidates, metric.shape[1], 3)
        integration_history = None
        if history:
            integration_history = torch.stack(history, dim=1).reshape(
                batch, candidates, len(history), metric.shape[1], 3
            )
        diagnostics = {}
        if feasibility_history:
            diagnostics["feasibility_cost_history"] = torch.stack(
                feasibility_history, dim=1
            ).reshape(batch, candidates, len(feasibility_history))
        if refinement_history:
            diagnostics["refinement_cost_history"] = torch.stack(
                refinement_history, dim=1
            ).reshape(batch, candidates, len(refinement_history))
        return TrajectoryBatch(
            trajectories,
            scores,
            integration_history,
            diagnostics=diagnostics or None,
        )


FlowPlanner = LegacyGuidedFlowPlanner
FlowPlannerConfig = LegacyFlowPlannerConfig
