import numpy as np

from emdp.examples import build_SB_example35, build_four_rooms_example
from emdp.examples import tricky_gridworlds
from emdp.examples import build_cake_world_mdp
from emdp import actions
from emdp import analytic

def test_SB_example35():

    mdp = build_SB_example35()

    mdp.set_current_state_to((0, 0))
    state, reward, done, _ = mdp.step(actions.UP)
    assert not done
    assert reward == -1
    assert mdp.unflatten_state(state) == (0, 0)

    state, reward, done, _ = mdp.step(actions.RIGHT)
    assert not done
    assert reward == 0
    assert mdp.unflatten_state(state) == (0, 1)

    state, reward, done, _ = mdp.step(actions.RIGHT)
    assert not done
    assert reward == +10
    assert mdp.unflatten_state(state) == (4, 1)


def test_four_rooms_loads():
    assert build_four_rooms_example() is not None


def test_make_multi_minima_reward_env():

    world, reward_spec = tricky_gridworlds.make_multi_minima_reward_env(
        best_reward=1, size=4)

    assert set(reward_spec.keys()) == {(3,0), (2,1), (1,2), (0, 3)}, \
        'Rewards must be along the diagonal'

    assert reward_spec[(3,0)] == 1
    assert reward_spec[(2,1)] == 3/4
    assert reward_spec[(1,2)] == 2/4
    assert reward_spec[(0,3)] == 1/4


def test_make_symmetric_epsilon_reward_env():

    world, reward_spec = tricky_gridworlds.make_symmetric_epsilon_reward_env(
        epsilon=0.5, size=4)

    assert set(reward_spec.keys()) == {(3,0), (0,3)}, 'Rewards must be on the corners'
    assert reward_spec[(3,0)] == 2.5
    assert reward_spec[(0,3)] == 5

def test_four_minima_env():
    world, reward_spec = tricky_gridworlds.make_four_minima_env(
        epsilon=0.5, size=5, best_reward=1)

    assert set(reward_spec.keys()) == {(2, 0),
                                       (0, 2),
                                       (2, 4),
                                       (4, 2)}, 'Rewards must be on edges.'
    assert reward_spec[(2, 0)] == 0.5
    assert reward_spec[(0, 2)] == 1*(1.5)
    assert reward_spec[(2, 4)] == 0.5
    assert reward_spec[(4, 2)] == 1
    assert world.p0.argmax() == 12 # Middle of the array.

def test_cakeworld_mdp():
    """Numerical test for cake world mdp.

    This numerical test ensures that calculations from implemented MDP represent
    those that are obtained from calculations in the paper.
    """
    epsilon = 0.1
    discount = 0.99
    built_mdp = build_cake_world_mdp(epsilon=0.1, discount=0.99)
    eval_policy = np.array([[0.5, 0.5], [0.5, 0.5]])

    calc_v_pi = analytic.calculate_V_pi(
        built_mdp.P, built_mdp.R, eval_policy, built_mdp.gamma)

    # Value of "Bad State" is independent of policy.
    expected_value = -2.0 * (1 + epsilon) / discount
    calculated_value = calc_v_pi[1]
    assert np.isclose(calculated_value, expected_value)
