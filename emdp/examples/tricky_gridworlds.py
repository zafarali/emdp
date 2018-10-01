"""
Environments with Tricky Rewards
Two kinds of worlds are available:
    1. Symmetric Grid World
        (0, size-1): +true_reward
        (size-1, 0): epsilon * true_reward
    2. Multi-minima Grid World
        (2,0): best_reward
        (1,1): best_reward*2/3
        (0,2): best_reward*1/3
"""
import numpy as np
from emdp.gridworld.builder_tools import build_simple_grid_world_with_terminal_states

GRID_SIZE = 10

def make_symmetric_epsilon_reward_env(epsilon, best_reward=+5,
                                      size=GRID_SIZE, p_success=1,
                                      gamma=0.99, seed=2017):
    """
    Symmetric Grid World where the rewards are at [(x,y): reward]:
        (0, size-1): +true_reward
        (size-1, 0): epsilon * true_reward

    :param epsilon: the proportion of the true reward to place
    :param best_reward: the true reward
    :param reward_spec: dict with the reward specification {(x,y):reward, ...}
    :param size: size of the grid world
    :param p_success: the probability an action is successful
    :param gamma: the discount factor for the MDP
    :param seed: the seed for the MDP
    :return:
    """
    reward_spec = {
        (0, size-1): best_reward,
        (size-1, 0): epsilon * best_reward
    }
    world = build_simple_grid_world_with_terminal_states(reward_spec,
                                                         size=size,
                                                         gamma=gamma,
                                                         p_success=p_success,
                                                         seed=seed)

    return world, reward_spec

def make_multi_minima_reward_env(best_reward=5, size=GRID_SIZE,
                                 p_success=1, gamma=0.99, shuffle_rewards=False,
                                 seed=2017):
    """
    Multiple minima grid world where there are rewards on the diagonal with increasing
    value. For example, in a 3x3 grid world we have:
        (2,0): best_reward
        (1,1): best_reward*2/3
        (0,2): best_reward*1/3

    if shuffle_rewards is True, we jumble the rewards along the diagonal.

    :param epsilon: the proportion of the true reward to place
    :param best_reward: the true reward
    :param reward_spec: dict with the reward specification {(x,y):reward, ...}
    :param size: size of the grid world
    :param p_success: the probability an action is successful
    :param gamma: the discount factor for the MDP
    :param shuffle_rewards: shuffle the rewards on the diagonal.
    :param seed: the seed for the MDP
    """

    all_rewards = [best_reward * (i + 1) / size for i in range(size)]

    if shuffle_rewards:
        randomizer = np.random.RandomState(seed) # reproducibility measure
        randomizer.shuffle(all_rewards)

    reward_spec = {(i,size-i-1): all_rewards[i] for i in range(size)}

    world = build_simple_grid_world_with_terminal_states(reward_spec,                                                         size=size,
                                                         gamma=gamma,
                                                         p_success=p_success,
                                                         seed=seed)

    return world, reward_spec


def make_four_minima_env(epsilon, best_reward=5, size=GRID_SIZE,
                         p_success=1, gamma=0.99, seed=2017):
    """
    Makes a gridworld where there are four rewards:
        {(0, size/2): (2 - epsilon)*best_reward,
         (size/2, 0): epsilon * best_reward,
         (size/2, size-1): epsilon * best_reward,
         (size-1, size/2): best_reward
        }
        
    and the agent starts in the middle of the grid at (size/2, size/2).
    Note that size must have an odd shape.
    :param best_reward: 
    :param size: 
    :param p_success: 
    :param gamma: 
    :param seed:
    :return: 
    """
    if size % 2 == 0: raise ValueError('grid must have an odd size.')
    middle = size//2
    end = size-1
    reward_spec = {
        (0, middle): (2-epsilon) * best_reward,
        (middle, 0): epsilon * best_reward,
        (middle, end): epsilon * best_reward,
        (end, middle): best_reward
    }
    start_state = middle * size + middle
    world = build_simple_grid_world_with_terminal_states(reward_spec,
                                                         size=size,
                                                         gamma=gamma,
                                                         p_success=p_success,
                                                         seed=seed,
                                                         start_state=start_state)

    return world, reward_spec
