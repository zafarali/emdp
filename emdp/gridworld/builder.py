"""
Utilities to help build more complex grid worlds.
"""
from .helper_utilities import (build_simple_grid,
                               get_possible_actions,
                               check_can_take_action,
                               get_state_after_executing_action)



class Builder(object):
    """
    Builder object to build a transition matrix for a grid world
    """

    def __init__(self, grid_size):
        pass

    def add_reward_at(self, tuple_location):
        pass

    def add_block_at(self, tuple_location):
        # how to add a wall:

        # find all the ways to go to "target_state"
        target_state = 0
        # from_states contains states that can lead you to target_state by executing from_action
        from_states, from_actions = np.where(sg[:, :, target_state] != 0)


        sg[from_states, from_actions, target_state] = 0

        # renormalize
        stacked = np.stack([(1 / sg.sum(2))] * 25)  # normalizing factor for each (from, action)
        (sg / np.transpose(stacked, (0, 2, 1))).sum(2) # should be 1
        pass

    def create_wall_between(self, tuple_location_start, tuple_location_end):
        pass