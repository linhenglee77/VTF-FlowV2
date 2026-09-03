"""Shared receding-horizon geometry and kinematic-bicycle execution."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BicycleConfig:
    wheelbase_m: float = 2.2
    control_dt_s: float = 0.1
    replan_interval_s: float = 0.5
    maximum_steering_deg: float = 30.0
    maximum_steering_rate_deg_s: float = 45.0
    maximum_acceleration_mps2: float = 1.5
    maximum_deceleration_mps2: float = 2.0
    maximum_speed_mps: float = 3.0
    lookahead_minimum_m: float = 1.25

    @property
    def controls_per_plan(self) -> int:
        ratio = self.replan_interval_s / self.control_dt_s
        rounded = int(round(ratio))
        if abs(ratio - rounded) > 1e-6:
            raise ValueError("replan interval must be an integer multiple of control dt")
        return rounded


def world_goal_to_local(goal: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Transform [X,Y,Z] goal into the current ego frame [forward,lateral,Z]."""

    dx, dy = goal[..., 0] - state[..., 0], goal[..., 1] - state[..., 1]
    cosine, sine = np.cos(state[..., 2]), np.sin(state[..., 2])
    return np.stack(
        [cosine * dx + sine * dy, -sine * dx + cosine * dy, goal[..., 2]], axis=-1
    ).astype(np.float32)


def local_path_to_world(path: np.ndarray, state: np.ndarray) -> np.ndarray:
    cosine, sine = math.cos(float(state[2])), math.sin(float(state[2]))
    result = np.array(path, dtype=np.float32, copy=True)
    result[:, 0] = state[0] + cosine * path[:, 0] - sine * path[:, 1]
    result[:, 1] = state[1] + sine * path[:, 0] + cosine * path[:, 1]
    return result


def warp_local_bev(
    initial_bev: torch.Tensor,
    pose: torch.Tensor,
    forward_m: float = 24.0,
    lateral_m: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-render the static initial-frame BEV in each current ego frame.

    Unknown cells outside the initial field of view are marked non-traversable
    and occupied. The returned known mask allows that assumption to be audited.
    """

    batch, _, height, width = initial_bev.shape
    forward = torch.linspace(
        forward_m / (2 * height), forward_m - forward_m / (2 * height),
        height, device=initial_bev.device, dtype=initial_bev.dtype,
    )
    lateral = torch.linspace(
        -lateral_m + lateral_m / width, lateral_m - lateral_m / width,
        width, device=initial_bev.device, dtype=initial_bev.dtype,
    )
    local_x, local_y = torch.meshgrid(forward, lateral, indexing="ij")
    local_x = local_x.unsqueeze(0).expand(batch, -1, -1)
    local_y = local_y.unsqueeze(0).expand(batch, -1, -1)
    cosine = torch.cos(pose[:, 2])[:, None, None]
    sine = torch.sin(pose[:, 2])[:, None, None]
    world_x = pose[:, 0, None, None] + cosine * local_x - sine * local_y
    world_y = pose[:, 1, None, None] + sine * local_x + cosine * local_y
    grid = torch.stack(
        [world_y / lateral_m, world_x / (forward_m / 2.0) - 1.0], dim=-1
    )
    warped = F.grid_sample(
        initial_bev, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    known = F.grid_sample(
        torch.ones(batch, 1, height, width, device=initial_bev.device, dtype=initial_bev.dtype),
        grid, mode="bilinear", padding_mode="zeros", align_corners=False,
    ).clamp(0.0, 1.0)
    warped[:, 0:1] *= known
    warped[:, 1:2] = warped[:, 1:2] * known + (1.0 - known)
    warped[:, 2:3] *= known
    return warped, known


def pure_pursuit_control(
    state: np.ndarray,
    path_world: np.ndarray,
    goal_world: np.ndarray,
    previous_steering: float,
    config: BicycleConfig,
) -> tuple[float, float]:
    xy = state[:2]
    nearest = int(np.argmin(np.linalg.norm(path_world[:, :2] - xy, axis=1)))
    lookahead = max(config.lookahead_minimum_m, float(state[3]) * 0.8)
    target_index = nearest
    while target_index + 1 < len(path_world):
        if np.linalg.norm(path_world[target_index, :2] - xy) >= lookahead:
            break
        target_index += 1
    target = path_world[target_index, :2]
    alpha = math.atan2(target[1] - state[1], target[0] - state[0]) - state[2]
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    requested = math.atan2(2.0 * config.wheelbase_m * math.sin(alpha), lookahead)
    maximum = math.radians(config.maximum_steering_deg)
    rate_step = math.radians(config.maximum_steering_rate_deg_s) * config.control_dt_s
    steering = float(np.clip(requested, previous_steering - rate_step, previous_steering + rate_step))
    steering = float(np.clip(steering, -maximum, maximum))
    goal_distance = float(np.linalg.norm(goal_world[:2] - state[:2]))
    desired_speed = float(np.clip(goal_distance / 3.0, 0.3, config.maximum_speed_mps))
    acceleration = float(np.clip(
        1.2 * (desired_speed - state[3]),
        -config.maximum_deceleration_mps2,
        config.maximum_acceleration_mps2,
    ))
    return steering, acceleration


def bicycle_step(
    state: np.ndarray, steering: float, acceleration: float, config: BicycleConfig
) -> np.ndarray:
    x, y, yaw, speed = [float(value) for value in state]
    speed = float(np.clip(
        speed + acceleration * config.control_dt_s, 0.0, config.maximum_speed_mps
    ))
    x += speed * math.cos(yaw) * config.control_dt_s
    y += speed * math.sin(yaw) * config.control_dt_s
    yaw += speed / config.wheelbase_m * math.tan(steering) * config.control_dt_s
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return np.asarray([x, y, yaw, speed], dtype=np.float32)
