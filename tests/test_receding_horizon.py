import numpy as np
import torch

from TerraFlow.closed_loop.receding_horizon import (
    BicycleConfig,
    bicycle_step,
    local_path_to_world,
    warp_local_bev,
    world_goal_to_local,
)


def test_receding_horizon_transforms_and_dynamics():
    state = np.asarray([2.0, 1.0, np.pi / 2, 0.0], dtype=np.float32)
    goal = np.asarray([2.0, 6.0, 0.5], dtype=np.float32)
    local = world_goal_to_local(goal, state)
    assert np.allclose(local, [5.0, 0.0, 0.5], atol=1e-5)
    world = local_path_to_world(np.asarray([[5.0, 0.0, 0.5]], dtype=np.float32), state)
    assert np.allclose(world[0], goal, atol=1e-5)
    bev = torch.zeros(1, 3, 64, 64)
    bev[:, 0] = 1.0
    warped, known = warp_local_bev(bev, torch.zeros(1, 4))
    assert warped.shape == bev.shape and known.shape == (1, 1, 64, 64)
    assert float(known.mean()) > 0.95
    next_state = bicycle_step(np.zeros(4, dtype=np.float32), 0.0, 1.0, BicycleConfig())
    assert next_state[0] > 0 and next_state[1] == 0
