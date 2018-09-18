import gym
import emdp.emdp_gym

def test_gym_registered():
    gym.make('sb-example3-v0')
    gym.make('two-state-v0')

def check_gym_step(env):
    env.step(env.action_space.sample())

def check_gym_reset(env):
    env.reset()


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