"""Typed, JSON-backed experiment configuration.

JSON is used for the initial configuration format so loading has no dependency
beyond Python and PyTorch. New sections can be added without embedding research
hyperparameters in model or dataset code.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset and trajectory-window settings.

    ``root`` deliberately defaults to ``None``. Users must supply their local
    RELLIS-3D location in an experiment config or at runtime.
    """

    root: Optional[str] = None
    history_steps: int = 10
    future_steps: int = 20
    point_features: int = 4

    def __post_init__(self) -> None:
        if self.history_steps <= 0 or self.future_steps <= 0:
            raise ValueError("history_steps and future_steps must be positive")
        if self.point_features < 3:
            raise ValueError("point_features must include at least xyz")


@dataclass(frozen=True)
class TerrainConfig:
    """Terrain representation settings."""

    representation: str = "bev_grid"
    resolution_m: float = 0.25
    extent_m: float = 40.0
    feature_dim: int = 64

    def __post_init__(self) -> None:
        if self.resolution_m <= 0.0 or self.extent_m <= 0.0:
            raise ValueError("terrain resolution and extent must be positive")
        if self.feature_dim <= 0:
            raise ValueError("terrain feature_dim must be positive")


@dataclass(frozen=True)
class PlannerConfig:
    """Trajectory sampling settings shared by future planner implementations."""

    horizon: int = 20
    trajectory_dim: int = 3
    num_samples: int = 16

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.num_samples <= 0:
            raise ValueError("planner horizon and num_samples must be positive")
        if self.trajectory_dim != 3:
            raise ValueError("version one requires trajectory_dim = 3 for xyz")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level VTF-Flow configuration."""

    seed: int = 7
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    def __post_init__(self) -> None:
        if self.dataset.future_steps != self.planner.horizon:
            raise ValueError(
                "dataset.future_steps must equal planner.horizon so supervision "
                "and predictions share H"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the immutable configuration to plain serializable values."""

        return asdict(self)


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"configuration section '{section}' must be an object")
    return value


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from a JSON file.

    Args:
        path: Path to a JSON configuration. Relative paths are resolved by the
            caller's process, never relative to a hard-coded dataset location.

    Returns:
        A validated immutable experiment configuration.
    """

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    root = _mapping(raw, "root")

    allowed = {"seed", "dataset", "terrain", "planner"}
    unknown = set(root) - allowed
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")

    return ExperimentConfig(
        seed=int(root.get("seed", 7)),
        dataset=DatasetConfig(**dict(_mapping(root.get("dataset", {}), "dataset"))),
        terrain=TerrainConfig(**dict(_mapping(root.get("terrain", {}), "terrain"))),
        planner=PlannerConfig(**dict(_mapping(root.get("planner", {}), "planner"))),
    )
