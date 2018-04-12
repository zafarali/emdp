"""
A simple grid world environment
"""
import numpy as np
import random
from ..common import Env, EpisodeDoneError, InvalidActionError

## GRID WORLDS HAVE 4 ACTIONS
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


POS_NEG = {(0, 3): -5, (3, 0): +5}
POS_POS = {(0, 3): +4, (3, 0): +5}
THREE_ASYM = {(0,3): +4, (3,0): +3, (3,3):+5}
THREE_SYM = {(0,3): +4, (3,0): +4, (3,3):+5}
THREE_SAME = {(0,3): +5, (3,0): +5, (3,3):+5}
THREE_LARGE = {(0,3): +5, (3,0): -5, (3,3):0}
AVOID_NEG = {(3,3):+5, (1,1): -5, (1, 2):-5, (2,1):-5}
AVOID_POS = {(3,3):+5, (1,1): +4, (1, 2):-5, (2,1):-5}

class SimpleGridWorld(Env):
    def __init__(self,
                 size=4,
                 seed=1337):
        super().__init__(seed)
        self.size = size
        self.start_location = None
        self.agent_location = None
        self.reset()

    def reset(self):
        """
        Resets the game
        """
        self.agent_location = random.choice(self.start_location)
        self.done = False
        return self.flatten_state(self.agent_location)

    def reset_to_location(self, state):
        """
        Resets the game to be in a specific location
        :param state:
        :return:
        """
        self.agent_location = state
        self.done = False


    class SimpleGridWorld(object):
        def __init__(self,
                     size=4,
                     reward_specs=POS_NEG,
                     start_loc=[(0, 0)]):
            """
            A simple grid world size x size grid world
            :param size: size of the gridworld
            :param reward_specs: the specification of the reward locations
            :param start_loc: the starting location
            """
            self.size = size
            self.reward_specs = reward_specs
            self.start_loc = start_loc
            self.agent_location = None
            self.done = True

        def reset(self):
            """
            Resets the game
            """
            self.agent_location = random.choice(self.start_loc)
            self.done = False
            return self.flatten_state(self.agent_location)

        def flatten_state(self, state):
            """Flatten state (x,y) into a one hot vector"""
            idx = self.size * state[0] + state[1]
            one_hot = np.zeros(self.size * self.size)
            one_hot[idx] = 1
            return one_hot

        def unflatten_state(self, onehot):
            onehot = onehot.reshape(self.size, self.size)
            x = onehot.argmax(0).max()
            y = onehot.argmax(1).max()
            return (x, y)

        def reset_to_location(self, state):
            """
            Resets the game to be in a specific location
            :param state:
            :return:
            """
            self.agent_location = state
            self.done = False

        def step(self, action):
            """
            Takes a step in the environment
            :param action: is the number corresponding to
            the action to take as follows:
                LEFT = 0
                RIGHT = 1
                UP = 2
                DOWN = 3
            """

            if self.done:
                raise EpisodeDoneError('The episode is done.')

            state = self.agent_location
            state_t = None
            if action == UP:
                if state[0] > 0:
                    # going "up" involves -1 in the x direction
                    state_t = (state[0] - 1, state[1])
            elif action == DOWN:
                if state[0] < self.size-1:
                    # going "down" involves going +1 in the x direction
                    state_t = (state[0] + 1, state[1])
            elif action == LEFT:
                if state[1] > 0:
                    # going "left" involves going -1 in the x direction
                    state_t = (state[0], state[1] - 1)

            elif action == RIGHT:
                if state[1] < self.size-1:
                    # going right involves going +1 in the x direction
                    state_t = (state[0], state[1] + 1)
            else:
                raise InvalidActionError('Unknown Action!')

            if state_t is None:
                state_t = (state[0], state[1])

            self.agent_location = state_t

            try:
                reward = self.reward_specs[state_t]
                self.done = True
            except KeyError as _:
                # no reward found!
                reward = 0
                self.done = False

            return self.flatten_state(state_t), reward, self.done, {}
