import torch

from TerraFlow.terrain.learned_feasibility_field import (
    FeasibilityFieldNet,
    LearnedTerrainField,
)


def test_learned_field_is_continuous_differentiable_and_speed_conditioned():
    torch.manual_seed(5)
    terrain = torch.rand(2, 3, 32, 32)
    model = FeasibilityFieldNet(width=8).eval()
    field = LearnedTerrainField(terrain, model)
    points = torch.tensor(
        [[[4.0, -1.0, 0.0], [8.0, 1.0, 0.0]], [[5.0, 0.0, 0.0], [9.0, 2.0, 0.0]]],
        requires_grad=True,
    )
    slow = field.cost(points, {"speed": torch.zeros(2, 2)})
    fast = field.cost(points, {"speed": torch.full((2, 2), 3.0)})
    assert slow.shape == (2, 2)
    assert torch.all(fast >= slow)
    slow.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    repeated = field.repeat_interleave(3)
    repeated_points = points.detach().repeat_interleave(3, dim=0)
    assert repeated.cost(repeated_points).shape == (6, 2)
