"""Construct timestamp-sampled ego-centric future trajectories from poses.

This module implements the requested relation

``inv(T_world_ego(t)) @ T_world_ego(t+i)``

and extracts its translation. The local RELLIS-3D audit could not verify pose
axis meanings or transform direction, so raw pose convention handling is kept in
the single :func:`transform_pose_convention` boundary. Its default is an
explicit no-op and must not be interpreted as confirmation of axis semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


CAMERA_FRAME_PATTERN = re.compile(
    r"^frame(?P<index>\d+)-(?P<seconds>\d+)_(?P<fraction>\d+)\.(?:jpg|jpeg)$",
    flags=re.IGNORECASE,
)
PoseConventionAdapter = Callable[[Tensor], Tensor]


def rellis3d_os1_to_planning_ego(raw_world_lidar: Tensor) -> Tensor:
    """Convert released Ouster poses to VTF-Flow's planning-ego basis.

    The local RELLIS-3D TF/pose audit verifies a 180-degree yaw between the
    Ouster LiDAR axes and the base-link-aligned planning axes. The planning ego
    deliberately retains the LiDAR origin to match the trajectory cache, so
    this adapter right-multiplies every ``T_world_lidar`` by
    ``diag(-1,-1,1,1)`` and introduces no translation.
    """

    if raw_world_lidar.shape[-2:] != (4, 4):
        raise TrajectoryConstructionError(
            f"pose transforms must end in [4, 4], got {tuple(raw_world_lidar.shape)}"
        )
    basis = torch.diag(
        torch.tensor(
            [-1.0, -1.0, 1.0, 1.0],
            device=raw_world_lidar.device,
            dtype=raw_world_lidar.dtype,
        )
    )
    return raw_world_lidar @ basis


class TrajectoryConstructionError(ValueError):
    """Raised when poses or timestamps cannot produce a requested trajectory."""


class TrajectoryValidationError(TrajectoryConstructionError):
    """Raised when a constructed trajectory violates a validity check."""


@dataclass(frozen=True)
class TrajectoryBuilderConfig:
    """Configuration for future trajectory construction.

    Attributes:
        horizon_seconds: Duration from the current pose to the final sample.
        sampling_interval_seconds: Time between requested future samples.
        min_future_frames: Minimum number of raw pose frames after the current
            frame. This guards against building from very short fragments.
        interpolate: Linearly interpolate world translations at exact requested
            times. When false, use the nearest timestamped future pose.
        max_speed_mps: Maximum accepted segment speed in output coordinates.
            The physical meaning of this threshold depends on verified pose
            units and must be configured accordingly.
        max_teleport_distance: Maximum accepted distance between adjacent output
            samples, including origin to the first future sample.
        origin_tolerance: Maximum numerical error allowed for
            ``inv(T_current) @ T_current`` translation.
    """

    horizon_seconds: float = 5.0
    sampling_interval_seconds: float = 0.5
    min_future_frames: int = 10
    interpolate: bool = True
    max_speed_mps: float = 30.0
    max_teleport_distance: float = 20.0
    origin_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.horizon_seconds <= 0.0:
            raise ValueError("horizon_seconds must be positive")
        if self.sampling_interval_seconds <= 0.0:
            raise ValueError("sampling_interval_seconds must be positive")
        ratio = self.horizon_seconds / self.sampling_interval_seconds
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(
                "horizon_seconds must be an integer multiple of "
                "sampling_interval_seconds"
            )
        if self.min_future_frames <= 0:
            raise ValueError("min_future_frames must be positive")
        if self.max_speed_mps <= 0.0 or self.max_teleport_distance <= 0.0:
            raise ValueError("speed and teleport thresholds must be positive")
        if self.origin_tolerance < 0.0:
            raise ValueError("origin_tolerance must be non-negative")

    @property
    def num_future_steps(self) -> int:
        """Number ``H`` of future samples; defaults to 10."""

        return int(round(self.horizon_seconds / self.sampling_interval_seconds))


@dataclass(frozen=True)
class TrajectoryValidity:
    """Results of all trajectory validity checks."""

    finite: bool
    timestamps_monotonic: bool
    origin_near_zero: bool
    no_teleportation: bool
    speed_within_limit: bool
    max_step_distance: float
    max_speed: float
    messages: Tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Whether every validity check passed."""

        return (
            self.finite
            and self.timestamps_monotonic
            and self.origin_near_zero
            and self.no_teleportation
            and self.speed_within_limit
        )


@dataclass(frozen=True)
class EgoTrajectory:
    """One ego-centric future trajectory and its sampling provenance."""

    xyz: Tensor
    current_origin: Tensor
    target_timestamps: Tensor
    source_frame_indices: Tensor
    current_frame_index: int
    validity: TrajectoryValidity


@dataclass(frozen=True)
class RellisSequence:
    """Pose and timestamp tensors loaded from one extracted sequence."""

    sequence_id: str
    sequence_dir: Path
    poses: Tensor
    timestamps: Tensor


def transform_pose_convention(
    raw_world_ego: Tensor,
    adapter: Optional[PoseConventionAdapter] = None,
) -> Tensor:
    """Apply the sole coordinate-convention adaptation boundary.

    Args:
        raw_world_ego: One transform ``[4, 4]`` or a batch ``[..., 4, 4]``.
            The caller supplies the semantic contract that these are
            ``T_world_ego`` transforms.
        adapter: Optional verified dataset-specific conversion. ``None`` keeps
            the numeric matrices unchanged. No axis swap or sign correction is
            built into VTF-Flow while the RELLIS convention remains unresolved.

    Returns:
        Transforms with the same shape, ready for relative-pose algebra.
    """

    if raw_world_ego.shape[-2:] != (4, 4):
        raise TrajectoryConstructionError(
            f"pose transforms must end in [4, 4], got {tuple(raw_world_ego.shape)}"
        )
    converted = raw_world_ego.clone() if adapter is None else adapter(raw_world_ego)
    if converted.shape != raw_world_ego.shape:
        raise TrajectoryConstructionError(
            "pose convention adapter must preserve the input tensor shape"
        )
    if not torch.isfinite(converted).all():
        raise TrajectoryConstructionError("pose convention output contains non-finite values")
    return converted


def relative_future_translations(
    current_world_ego: Tensor,
    future_world_ego: Tensor,
    convention_adapter: Optional[PoseConventionAdapter] = None,
) -> Tuple[Tensor, Tensor]:
    """Apply the exact relative-transform formula and extract xyz translation.

    Returns:
        A tuple ``(future_xyz, current_origin)``. ``future_xyz`` has shape
        ``[H, 3]`` and ``current_origin`` is the translation obtained from
        ``inv(T_current) @ T_current``.
    """

    if current_world_ego.shape != (4, 4):
        raise TrajectoryConstructionError("current_world_ego must have shape [4, 4]")
    if future_world_ego.ndim != 3 or future_world_ego.shape[-2:] != (4, 4):
        raise TrajectoryConstructionError("future_world_ego must have shape [H, 4, 4]")

    all_poses = torch.cat((current_world_ego.unsqueeze(0), future_world_ego), dim=0)
    converted = transform_pose_convention(all_poses, convention_adapter)
    current = converted[0]
    future = converted[1:]
    try:
        current_inverse = torch.linalg.inv(current)
    except RuntimeError as error:
        raise TrajectoryConstructionError("current pose is not invertible") from error
    local_future = current_inverse.unsqueeze(0) @ future
    local_current = current_inverse @ current
    return local_future[:, :3, 3], local_current[:3, 3]


def validate_trajectory(
    xyz: Tensor,
    current_origin: Tensor,
    timestamps: Tensor,
    config: TrajectoryBuilderConfig,
) -> TrajectoryValidity:
    """Check numerical, temporal, displacement, and speed validity.

    ``timestamps`` includes the current timestamp followed by all future target
    timestamps and therefore has length ``H + 1``.
    """

    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise TrajectoryConstructionError("xyz must have shape [H, 3]")
    if current_origin.shape != (3,):
        raise TrajectoryConstructionError("current_origin must have shape [3]")
    if timestamps.ndim != 1 or timestamps.numel() != xyz.shape[0] + 1:
        raise TrajectoryConstructionError("timestamps must have shape [H + 1]")

    finite = bool(
        torch.isfinite(xyz).all()
        and torch.isfinite(current_origin).all()
        and torch.isfinite(timestamps).all()
    )
    time_deltas = timestamps[1:] - timestamps[:-1]
    timestamps_monotonic = bool(torch.isfinite(time_deltas).all() and (time_deltas > 0).all())
    origin_error = float(torch.linalg.vector_norm(current_origin)) if finite else float("inf")
    origin_near_zero = finite and origin_error <= config.origin_tolerance

    if finite and timestamps_monotonic:
        points = torch.cat((current_origin.unsqueeze(0), xyz), dim=0)
        distances = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)
        speeds = distances / time_deltas
        max_step = float(distances.max()) if distances.numel() else 0.0
        max_speed = float(speeds.max()) if speeds.numel() else 0.0
    else:
        max_step = float("inf")
        max_speed = float("inf")

    no_teleportation = finite and max_step <= config.max_teleport_distance
    speed_within_limit = finite and timestamps_monotonic and max_speed <= config.max_speed_mps
    messages: List[str] = []
    if not finite:
        messages.append("trajectory, origin, or timestamps contain non-finite values")
    if not timestamps_monotonic:
        messages.append("timestamps are not strictly increasing")
    if not origin_near_zero:
        messages.append(
            f"current pose is {origin_error:.6g} from local origin "
            f"(limit {config.origin_tolerance:.6g})"
        )
    if not no_teleportation:
        messages.append(
            f"maximum step {max_step:.6g} exceeds teleport limit "
            f"{config.max_teleport_distance:.6g}"
        )
    if not speed_within_limit:
        messages.append(
            f"maximum speed {max_speed:.6g} exceeds limit {config.max_speed_mps:.6g}"
        )

    return TrajectoryValidity(
        finite=finite,
        timestamps_monotonic=timestamps_monotonic,
        origin_near_zero=origin_near_zero,
        no_teleportation=no_teleportation,
        speed_within_limit=speed_within_limit,
        max_step_distance=max_step,
        max_speed=max_speed,
        messages=tuple(messages),
    )


def _check_pose_and_timestamp_inputs(poses: Tensor, timestamps: Tensor) -> None:
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise TrajectoryConstructionError("poses must have shape [N, 4, 4]")
    if timestamps.ndim != 1 or timestamps.shape[0] != poses.shape[0]:
        raise TrajectoryConstructionError("timestamps must have shape [N] matching poses")
    if poses.shape[0] < 2:
        raise TrajectoryConstructionError("at least two poses are required")
    if not torch.isfinite(poses).all() or not torch.isfinite(timestamps).all():
        raise TrajectoryConstructionError("poses and timestamps must be finite")
    if not bool(((timestamps[1:] - timestamps[:-1]) > 0).all()):
        raise TrajectoryConstructionError("dataset timestamps must be strictly increasing")


def _sample_future_poses(
    poses: Tensor,
    timestamps: Tensor,
    target_timestamps: Tensor,
    interpolate: bool,
) -> Tuple[Tensor, Tensor]:
    """Sample future matrices, interpolating translations when requested."""

    right_indices = torch.searchsorted(timestamps, target_timestamps, right=False)
    if bool((right_indices >= timestamps.numel()).any()):
        raise TrajectoryConstructionError("requested horizon extends past available timestamps")

    if not interpolate:
        left_indices = torch.clamp(right_indices - 1, min=0)
        right_errors = torch.abs(timestamps[right_indices] - target_timestamps)
        left_errors = torch.abs(timestamps[left_indices] - target_timestamps)
        use_left = left_errors <= right_errors
        nearest = torch.where(use_left, left_indices, right_indices)
        return poses[nearest].clone(), nearest

    left_indices = torch.clamp(right_indices - 1, min=0)
    exact = timestamps[right_indices] == target_timestamps
    denominator = timestamps[right_indices] - timestamps[left_indices]
    safe_denominator = torch.where(exact, torch.ones_like(denominator), denominator)
    alpha = torch.where(
        exact,
        torch.ones_like(target_timestamps),
        (target_timestamps - timestamps[left_indices]) / safe_denominator,
    )
    sampled = poses[right_indices].clone()
    left_translation = poses[left_indices, :3, 3]
    right_translation = poses[right_indices, :3, 3]
    sampled[:, :3, 3] = torch.lerp(left_translation, right_translation, alpha.unsqueeze(-1))
    # Future rotation does not affect translation extracted from inv(T_current) @
    # T_future. The right-bracketing rotation is retained rather than inventing
    # an undocumented orientation interpolation convention.
    return sampled, right_indices


class RellisTrajectoryBuilder:
    """Build fixed-time future trajectories from a timestamped pose sequence."""

    def __init__(
        self,
        config: Optional[TrajectoryBuilderConfig] = None,
        convention_adapter: Optional[PoseConventionAdapter] = None,
    ) -> None:
        self.config = config or TrajectoryBuilderConfig()
        self.convention_adapter = convention_adapter

    def valid_current_indices(self, poses: Tensor, timestamps: Tensor) -> List[int]:
        """Return indices with enough raw frames and temporal horizon."""

        _check_pose_and_timestamp_inputs(poses, timestamps)
        config = self.config
        valid: List[int] = []
        for index in range(poses.shape[0] - 1):
            if poses.shape[0] - index - 1 < config.min_future_frames:
                continue
            if float(timestamps[-1] - timestamps[index]) + 1e-9 < config.horizon_seconds:
                continue
            valid.append(index)
        return valid

    def build(self, poses: Tensor, timestamps: Tensor, current_index: int) -> EgoTrajectory:
        """Build and validate one fixed-duration future trajectory."""

        _check_pose_and_timestamp_inputs(poses, timestamps)
        config = self.config
        if current_index < 0 or current_index >= poses.shape[0]:
            raise IndexError(f"current_index {current_index} is outside the pose sequence")
        future_frames = poses.shape[0] - current_index - 1
        if future_frames < config.min_future_frames:
            raise TrajectoryConstructionError(
                f"only {future_frames} future raw frames; "
                f"minimum is {config.min_future_frames}"
            )

        current_time = timestamps[current_index]
        offsets = torch.arange(
            1,
            config.num_future_steps + 1,
            device=timestamps.device,
            dtype=timestamps.dtype,
        ) * config.sampling_interval_seconds
        targets = current_time + offsets
        future_poses, source_indices = _sample_future_poses(
            poses, timestamps, targets, config.interpolate
        )
        xyz, current_origin = relative_future_translations(
            poses[current_index], future_poses, self.convention_adapter
        )
        validation_times = torch.cat((current_time.reshape(1), targets))
        validity = validate_trajectory(xyz, current_origin, validation_times, config)
        if not validity.is_valid:
            raise TrajectoryValidationError("; ".join(validity.messages))
        return EgoTrajectory(
            xyz=xyz,
            current_origin=current_origin,
            target_timestamps=targets,
            source_frame_indices=source_indices,
            current_frame_index=current_index,
            validity=validity,
        )


def load_pose_matrices(path: Path, dtype: torch.dtype = torch.float64) -> Tensor:
    """Load 12-value pose rows as homogeneous ``[N, 4, 4]`` matrices."""

    rows: List[List[float]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError as error:
            raise TrajectoryConstructionError(
                f"{path}:{line_number}: pose row contains non-numeric text"
            ) from error
        if len(values) != 12:
            raise TrajectoryConstructionError(
                f"{path}:{line_number}: expected 12 values, got {len(values)}"
            )
        rows.append(values)
    if not rows:
        raise TrajectoryConstructionError(f"no pose rows found in {path}")
    matrices = torch.eye(4, dtype=dtype).repeat(len(rows), 1, 1)
    matrices[:, :3, :] = torch.tensor(rows, dtype=dtype).reshape(-1, 3, 4)
    return matrices


def load_rgb_timestamps(
    image_dir: Path,
    expected_frames: int,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Load timestamps embedded in RGB names and align them by frame index."""

    by_index: Dict[int, float] = {}
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        match = CAMERA_FRAME_PATTERN.fullmatch(path.name)
        if not match:
            continue
        fraction = match.group("fraction")
        timestamp = int(match.group("seconds")) + int(fraction) / (10 ** len(fraction))
        index = int(match.group("index"))
        if index in by_index:
            raise TrajectoryConstructionError(f"duplicate RGB frame index {index} in {image_dir}")
        by_index[index] = timestamp
    missing = [index for index in range(expected_frames) if index not in by_index]
    if missing:
        preview = ", ".join(str(value) for value in missing[:8])
        raise TrajectoryConstructionError(
            f"RGB timestamps missing for {len(missing)} pose indices; examples: {preview}"
        )
    return torch.tensor([by_index[index] for index in range(expected_frames)], dtype=dtype)


def load_rellis_sequence(sequence_dir: Path) -> RellisSequence:
    """Load pose rows and index-matched RGB timestamps from one sequence."""

    sequence_path = sequence_dir.expanduser().resolve()
    poses = load_pose_matrices(sequence_path / "poses.txt")
    timestamps = load_rgb_timestamps(
        sequence_path / "pylon_camera_node", expected_frames=poses.shape[0]
    )
    _check_pose_and_timestamp_inputs(poses, timestamps)
    return RellisSequence(
        sequence_id=sequence_path.name,
        sequence_dir=sequence_path,
        poses=poses,
        timestamps=timestamps,
    )
