from emdp.examples import build_SB_example35
from emdp.examples import tricky_gridworlds
from emdp import actions

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