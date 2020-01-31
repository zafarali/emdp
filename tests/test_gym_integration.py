import numpy as np
import gym
import emdp.emdp_gym
from emdp import examples

def test_gym_registered():
    gym.make('sb-example3-v0')
    gym.make('two-state-v0')

def check_gym_step(env):
    assert env.step(env.action_space.sample()) is not None

def check_gym_reset(env):
    assert env.reset() is not None


def test_gym_step():
    env = gym.make('sb-example3-v0')
    check_gym_step(env)
    env = gym.make('two-state-v0')
    check_gym_step(env)

def test_gym_reset():
    env = gym.make('sb-example3-v0')
    check_gym_reset(env)
    env = gym.make('two-state-v0')
    check_gym_reset(env)

def test_gym_onehot_observation():
    env = gym.make('sb-example3-v0')
    state = env.reset()

    assert isinstance(state, np.ndarray)
    assert len(state.shape) == 1
    assert state.shape == (5*5, )
    # assert state.dtype == np.int8 Breaking change for a future version.

def test_gym_int_observation():
    env = emdp.emdp_gym.gymify(
        examples.build_SB_example35(),
        observation_one_hot=False)
    state = env.reset()
    assert type(state) == int
