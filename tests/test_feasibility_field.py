import torch

from TerraFlow.terrain.feasibility_field import AnalyticTerrainField


def test_obstacle_has_lower_feasibility_and_query_is_differentiable():
    terrain = torch.zeros(1, 3, 64, 64)
    terrain[:, 0] = 1.0
    terrain[:, 1, 32, 32] = 1.0
    field = AnalyticTerrainField(terrain)
    clear = torch.tensor([[[4.0, -8.0, 0.0]]], requires_grad=True)
    obstacle = torch.tensor([[[12.2, 0.2, 0.0]]], requires_grad=True)
    assert float(field.query(obstacle).detach()) < float(field.query(clear).detach())
    field.cost(obstacle).sum().backward()
    assert obstacle.grad is not None and torch.isfinite(obstacle.grad).all()
