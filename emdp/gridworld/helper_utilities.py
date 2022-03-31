import numpy as np
from ..actions import LEFT, RIGHT, UP, DOWN
from ..exceptions import InvalidActionError
from typing import List
n_actions = 4


def flatten_state(state, size, state_space):
    """Flatten state (x,y) into a one hot vector."""
    idx = size * state[0] + state[1]
    one_hot = np.zeros(state_space)
    one_hot[idx] = 1
    return one_hot


def unflatten_state(onehot, size, has_absorbing_state):
    """Unflatten a one hot vector into a (x,y) pair"""
    if has_absorbing_state:
        onehot = onehot[:-1]
    onehot = onehot.reshape(size, size)
    x = onehot.argmax(0).max()
    y = onehot.argmax(1).max()
    return (x, y)


def get_state_after_executing_action(action, state, grid_size):
    """
    Gets the state after executing an action
    
    :param action:
    :param state:
    :param grid_size:
    :return:
    """
    if check_can_take_action(action, state, grid_size):
        if action == LEFT:
            return state-1
        elif action == RIGHT:
            return state+1
        elif action == UP:
            return state - grid_size
        elif action == DOWN:
            return state + grid_size
    else:
        # cant execute action, stay in the same place.
        return state


def check_can_take_action(action, state, grid_size):
    """
    checks if you can take an action in a state.
    :param action:
    :param state:
    :param grid_size:
    :return:
    """
    LAST_ROW = list(range(grid_size*(grid_size-1), grid_size*grid_size))
    FIRST_ROW = list(range(0, grid_size))
    LEFT_EDGE = list(range(0, grid_size*grid_size, grid_size))
    RIGHT_EDGE = list(range(grid_size-1, grid_size*grid_size, grid_size))

    if action == DOWN:
        if state in LAST_ROW:
            return False
    elif action == RIGHT:
        if state in RIGHT_EDGE:
            return False
    elif action == UP:
        if state in FIRST_ROW:
            return False
    elif action == LEFT:
        if state in LEFT_EDGE:
            return False
    else:
        raise InvalidActionError('Cannot take action {} in a grid world of size {}x{}'.format(action, grid_size, grid_size))

    return True


def get_possible_actions(state, grid_size):
    """Gets all possible actions at a given state.


    Args:
        state (_type_): _description_
        grid_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    LAST_ROW = list(range(grid_size*(grid_size-1), grid_size*grid_size))
    FIRST_ROW = list(range(0, grid_size))
    LEFT_EDGE = list(range(0, grid_size*grid_size, grid_size))
    RIGHT_EDGE = list(range(grid_size-1, grid_size*grid_size, grid_size))

    available_actions = [LEFT, RIGHT, UP, DOWN]
    if state in LAST_ROW:
        available_actions.remove(DOWN)
    if state in FIRST_ROW:
        available_actions.remove(UP)
    if state in RIGHT_EDGE:
        available_actions.remove(RIGHT)
    if state in LEFT_EDGE:
        available_actions.remove(LEFT)
    return available_actions


# def flatten_state(state, n_states, grid_size):
#     """Flatten state (x,y) into a one hot vector"""
#     idx =
#     one_hot = np.zeros(n_states)
#     one_hot[idx] = 1
#     return one_hot

def build_simple_grid(size=5, terminal_states: List = None, p_success=1):
    """
    Builds a simple grid where an agent can move *LEFT*, *RIGHT*, *UP* or *DOWN*
    and actions success with probability ``p_success``.
    A terminal state is added if :code:`len(terminal_states) > 0` and will return matrix of
    size :math:`(|S|+1)\\times|A|\\times(|S|+1)`.

    Moving into walls does nothing.

    Examples:

        Builds a simple 5x5 grid world where there is a terminal state at (0, 4). 
        The probability of successfully executing the action is 0.9. 
        This function returns the transition matrix.

        >>> grid = build_simple_grid(size=5, terminal_states=[(0, 4)], p_success=0.9)
        >>> print(grid.shape)
        (26, 4, 26)

    Args:
        size (int, optional): size of the grid world. Defaults to 5.
            :math:`|S| = size \\times size`
        terminal_states (list, optional): the location of terminal states: a list of (x, y) tuples. Defaults to [].
        p_success (int, optional): the probabilty that an action will be successful. Defaults to 1.

    Raises:
        InvalidActionError

    Returns:
        np.ndarray: the transition matrix of the given grid world. shape: :math:`\left(|S|+1,|A|,|S|+1\\right)`
    """
    if terminal_states is None:
        terminal_states = []

    p_fail = 1 - p_success

    n_states = size*size
    grid_states = n_states  # the number of entries of the state vector
    # corresponding to the grid itself.
    if len(terminal_states) > 0:
        n_states += 1  # add an entry to state vector for terminal state
    terminal_states = list(map(lambda tupl: int(size * tupl[0] + tupl[1]), terminal_states))

    # this helper function creates the state transition list for
    # taking an action in a state
    def create_state_list_for_action(state_idx, action):
        transition_probs = np.zeros(n_states)
        if state_idx in terminal_states:
            # no matter what action you take you should go to the absorbing state
            transition_probs[-1] = 1
        elif state_idx == n_states-1 and len(terminal_states) > 0:
            # absorbing state, you should just transition back here whatever action you take.
            transition_probs[-1] = 1

        elif action in [LEFT, RIGHT, UP, DOWN]:
            # valid action, now see if we can actually execute this action
            # in this state:
            # TODO: distinguish between capability of slipping and taking wrong action vs failing to execute action.
            if check_can_take_action(action, state_idx, size):
                # yes we can
                possible_actions = get_possible_actions(state_idx, size)
                if action in possible_actions:
                    transition_probs[get_state_after_executing_action(action, state_idx, size)] = p_success
                    possible_actions.remove(action)
                for other_action in possible_actions:
                    transition_probs[get_state_after_executing_action(other_action, state_idx, size)] = p_fail/len(possible_actions)

            else:
                possible_actions = get_possible_actions(state_idx, size)
                transition_probs[state_idx] = p_success  # cant take action, stay in same place
                for other_action in possible_actions:
                    transition_probs[get_state_after_executing_action(other_action, state_idx, size)] = p_fail/len(possible_actions)

        else:
            raise InvalidActionError('Invalid action {} in the 2D gridworld'.format(action))
        return transition_probs

    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            P[s, a, :] = create_state_list_for_action(s, a)
    #
    # T = {s: {a: create_state_list_for_action(s, a) for a in range(n_actions)} for s in range(n_states)}
    # T[0][LEFT][0], T[0][RIGHT][0], T[0][DOWN][0], T[0][UP][0] = 1, 1, 1, 1
    # T[15][LEFT][15], T[15][RIGHT][15], T[15][DOWN][15], T[15][UP][15] = 1, 1, 1, 1
    return P


def add_walls():
    pass
