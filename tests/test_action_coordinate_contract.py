"""Regression tests for the solver-space versus physical-action-space contract.

These tests intentionally fail against upstream main (21446f1). The policy
currently passes raw environment bounds to a planner operating in transformed
action coordinates. A future fix may additionally impose a tighter model trust
region, but must still enforce physical feasibility after inverse_transform.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from stable_worldmodel.data.normalization import PercentileScaler, ZScoreScaler
from stable_worldmodel.planning.solver.icem import ICEMSolver
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy


@pytest.mark.parametrize('action_block', [1, 2])
@pytest.mark.parametrize('normalizer', ['zscore', 'percentile'])
def test_solver_limits_remain_physically_feasible_after_inverse_transform(
    action_block, normalizer
):
    """Every candidate inside a solver's Box must map into the raw env Box.

    Deliberately chosen fitted scalers make a numerical clamp of [-1, 1] in
    *model* coordinates insufficient to enforce the env's [-1, 1] action
    constraints. This test does not prescribe how wide the model trust region
    should be; a strict subset of feasible model actions is valid.
    """
    if normalizer == 'zscore':
        scaler = ZScoreScaler(
            mean=np.array([0.8, -0.8], dtype=np.float32),
            std=np.array([0.5, 0.5], dtype=np.float32),
        )
    else:
        scaler = PercentileScaler(
            q_low=np.array([-2.0, -2.0], dtype=np.float32),
            q_high=np.array([2.0, 2.0], dtype=np.float32),
        )

    raw_low = np.full((2, 2), -1.0, dtype=np.float32)
    raw_high = np.full((2, 2), 1.0, dtype=np.float32)
    env = SimpleNamespace(
        num_envs=2,
        action_space=Box(low=raw_low, high=raw_high, dtype=np.float32),
        single_action_space=Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    )

    solver = ICEMSolver(
        cost=None, batch_size=2, num_samples=8, n_steps=1, topk=2
    )
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=3, receding_horizon=1, action_block=action_block
        ),
        process={'action': scaler},
    )
    policy.set_env(env)

    # The solver's bounds are in the same coordinates as its world-model calls.
    model_low = solver._action_low.cpu().numpy().reshape(action_block, 2)
    model_high = solver._action_high.cpu().numpy().reshape(action_block, 2)
    physical_low = scaler.inverse_transform(model_low)
    physical_high = scaler.inverse_transform(model_high)

    np.testing.assert_array_less(
        np.full_like(physical_low, -1.0 - 1e-6), physical_low,
        err_msg=(
            f'Lower solver limit maps outside physical Box: {physical_low}'
        ),
    )
    np.testing.assert_array_less(
        physical_high, np.full_like(physical_high, 1.0 + 1e-6),
        err_msg=(
            f'Upper solver limit maps outside physical Box: {physical_high}'
        ),
    )


@pytest.mark.parametrize('action_block', [1, 2])
def test_explicit_model_trust_region_intersects_physical_limits(action_block):
    """Keep a model trust interval, but never at the cost of feasibility."""
    scaler = ZScoreScaler(
        mean=np.array([0.8, -0.8], dtype=np.float32),
        std=np.array([0.5, 0.5], dtype=np.float32),
    )
    env = SimpleNamespace(
        num_envs=2,
        action_space=Box(low=-1, high=1, shape=(2, 2)),
        single_action_space=Box(low=-1, high=1, shape=(2,)),
    )
    solver = ICEMSolver(cost=None, n_steps=1, num_samples=8, topk=2)
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=3,
            receding_horizon=1,
            action_block=action_block,
            model_action_bounds=(-1.0, 1.0),
        ),
        process={'action': scaler},
    )
    policy.set_env(env)
    expected_low = np.tile([-1.0, -0.4], action_block)
    expected_high = np.tile([0.4, 1.0], action_block)
    np.testing.assert_allclose(
        solver._action_low.cpu().numpy(), expected_low, atol=1e-6
    )
    np.testing.assert_allclose(
        solver._action_high.cpu().numpy(), expected_high, atol=1e-6
    )


def test_pusht_scale_can_retain_original_model_trust_interval():
    """Physical +-1 is wider than a model-space +-1 trust interval."""
    scaler = ZScoreScaler(mean=[0, 0], std=[0.21, 0.21])
    env = SimpleNamespace(
        num_envs=1,
        action_space=Box(low=-1, high=1, shape=(1, 2)),
        single_action_space=Box(low=-1, high=1, shape=(2,)),
    )
    solver = ICEMSolver(cost=None, n_steps=1, num_samples=8, topk=2)
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=3,
            receding_horizon=1,
            model_action_bounds=(-1.0, 1.0),
        ),
        process={'action': scaler},
    )
    policy.set_env(env)
    np.testing.assert_allclose(solver._action_low, [-1, -1])
    np.testing.assert_allclose(solver._action_high, [1, 1])


def test_percentile_scaler_rejects_unrepresentable_physical_box():
    """When percentile support is wholly outside the env Box, fail loudly."""
    scaler = PercentileScaler(q_low=[2, 2], q_high=[3, 3])
    env = SimpleNamespace(
        num_envs=1,
        action_space=Box(low=-1, high=1, shape=(1, 2)),
        single_action_space=Box(low=-1, high=1, shape=(2,)),
    )
    solver = ICEMSolver(cost=None, n_steps=1, num_samples=8, topk=2)
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(horizon=3, receding_horizon=1),
        process={'action': scaler},
    )
    with pytest.raises(ValueError, match='cannot represent physically'):
        policy.set_env(env)


def test_disjoint_model_trust_and_physical_box_rejected():
    scaler = ZScoreScaler(mean=[10, 10], std=[1, 1])
    env = SimpleNamespace(
        num_envs=1,
        action_space=Box(low=-1, high=1, shape=(1, 2)),
        single_action_space=Box(low=-1, high=1, shape=(2,)),
    )
    solver = ICEMSolver(cost=None, n_steps=1, num_samples=8, topk=2)
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=3,
            receding_horizon=1,
            model_action_bounds=(-1.0, 1.0),
        ),
        process={'action': scaler},
    )
    with pytest.raises(ValueError, match='empty intersection'):
        policy.set_env(env)


def test_icem_returned_ema_mean_obeys_bounds():
    """An initially infeasible mean can survive EMA despite clipped elites."""
    class ConstantCost:
        def get_cost(self, info_dict, action_candidates):
            return action_candidates.new_zeros(action_candidates.shape[:2])

    solver = ICEMSolver(
        cost=ConstantCost(),
        batch_size=1,
        num_samples=8,
        n_steps=1,
        topk=2,
        alpha=0.99,
        return_mean=True,
        seed=42,
    )
    model_box = Box(low=3.0, high=5.0, shape=(1, 2))
    solver.configure(
        action_space=model_box,
        n_envs=1,
        config=PlanConfig(horizon=1, receding_horizon=1),
    )
    actions = solver({'state': torch.zeros(1, 1, 2)})['actions']
    assert bool(torch.all(actions >= 3.0))
    assert bool(torch.all(actions <= 5.0))


def _two_action_env(num_envs=1):
    return SimpleNamespace(
        num_envs=num_envs,
        action_space=Box(low=-1, high=1, shape=(num_envs, 2)),
        single_action_space=Box(low=-1, high=1, shape=(2,)),
    )


def test_default_preserves_pusht_numerical_trust_region():
    """Default must not expand +-1 to +-4.76 after fixing coordinates."""
    scaler = ZScoreScaler(mean=[0, 0], std=[0.21, 0.21])
    solver = ICEMSolver(cost=None, n_steps=1, num_samples=8, topk=2)
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(horizon=3, receding_horizon=1),
        process={'action': scaler},
    )
    policy.set_env(_two_action_env())
    np.testing.assert_allclose(solver._action_low, [-1.0, -1.0])
    np.testing.assert_allclose(solver._action_high, [1.0, 1.0])


def test_physical_only_bounds_are_explicit_opt_in():
    scaler = ZScoreScaler(mean=[0, 0], std=[0.21, 0.21])
    solver = ICEMSolver(cost=None, n_steps=1, num_samples=8, topk=2)
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=3, receding_horizon=1, model_action_bounds=None
        ),
        process={'action': scaler},
    )
    policy.set_env(_two_action_env())
    np.testing.assert_allclose(solver._action_low, [-1 / 0.21] * 2, rtol=1e-6)
    np.testing.assert_allclose(solver._action_high, [1 / 0.21] * 2, rtol=1e-6)


@pytest.mark.parametrize('with_scaler', [True, False])
def test_heterogeneous_vector_env_bounds_fail_explicitly(with_scaler):
    process = (
        {'action': ZScoreScaler(mean=[0, 0], std=[0.5, 0.5])}
        if with_scaler else {}
    )
    env = SimpleNamespace(
        num_envs=2,
        action_space=Box(
            low=np.array([[-1, -1], [-0.5, -0.5]], dtype=np.float32),
            high=np.array([[1, 1], [0.5, 0.5]], dtype=np.float32),
        ),
        single_action_space=Box(low=-1, high=1, shape=(2,)),
    )
    solver = ICEMSolver(cost=None, n_steps=1, num_samples=8, topk=2)
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(horizon=3, receding_horizon=1),
        process=process,
    )
    with pytest.raises(ValueError, match='Heterogeneous per-environment'):
        policy.set_env(env)


def test_nonmonotone_inverse_scaler_is_rejected():
    """A negative-scale inverse must not pass endpoint feasibility checks."""
    scaler = ZScoreScaler(mean=[0, 0], std=[-0.5, -0.5])
    solver = ICEMSolver(cost=None, n_steps=1, num_samples=8, topk=2)
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(horizon=3, receding_horizon=1),
        process={'action': scaler},
    )
    with pytest.raises(ValueError, match='cannot represent physically'):
        policy.set_env(_two_action_env())


@pytest.mark.parametrize('action_block', [1, 2])
@pytest.mark.parametrize('normalizer', ['zscore', 'percentile'])
@pytest.mark.parametrize('num_envs', [1, 3])
def test_icem_policy_executed_actions_obey_raw_box(
    action_block, normalizer, num_envs
):
    """Exercise the real solver -> policy -> inverse scaler -> env path.

    The cost deliberately favors extreme candidate magnitudes. Check every
    executed step, including replanning with the previous plan as warm start.
    """

    class ExtremeActionCost:
        def get_cost(self, info_dict, action_candidates):
            return -action_candidates.square().sum(dim=(-1, -2))

    if normalizer == 'zscore':
        scaler = ZScoreScaler(
            mean=np.array([0.8, -0.8], dtype=np.float32),
            std=np.array([0.5, 0.5], dtype=np.float32),
        )
    else:
        scaler = PercentileScaler(
            q_low=np.array([-2.0, -2.0], dtype=np.float32),
            q_high=np.array([2.0, 2.0], dtype=np.float32),
        )

    env = SimpleNamespace(
        num_envs=num_envs,
        action_space=Box(
            low=-1.0, high=1.0, shape=(num_envs, 2), dtype=np.float32
        ),
        single_action_space=Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        ),
    )
    solver = ICEMSolver(
        cost=ExtremeActionCost(),
        batch_size=2,
        num_samples=32,
        n_steps=3,
        topk=6,
        seed=7,
    )
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=3,
            receding_horizon=1,
            action_block=action_block,
        ),
        process={'action': scaler},
    )
    policy.set_env(env)
    info = {'state': np.zeros((num_envs, 1, 2), dtype=np.float32)}

    for _ in range(2 * action_block + 1):
        action = policy.get_action(info.copy())
        assert action.shape == (num_envs, 2)
        assert np.isfinite(action).all()
        assert np.all(action >= env.action_space.low - 1e-6)
        assert np.all(action <= env.action_space.high + 1e-6)
