from emdp import build_chain_MDP
import numpy as np

def test_build_chain_MDP():
    mdp = build_chain_MDP(n_states=3, starting_distribution=np.array([0, 0, 1]),
                          terminal_states=[0], reward_spec=[(1, 0, +5)], p_success=0.9)
    """
    this MDP looks like this:
    [ 0 ] --> [ 0 ] with probability 1 for all actions
    [ 1 ] --> [ 2 ] with probability 0.9 if taking RIGHT
    [ 1 ] --> [ 0 ] with probability 0.9 if taking LEFT (also gets a reward of +1)
    [ 2 ] --> [ 2 ] with probability 1 if taking RIGHT
    [ 2 ] --> [ 1 ] with probability 0.9 if taking LEFT
    """
    assert mdp.P[0][0][0] == 1 and mdp.P[0][1][0] == 1, 'terminal state is non absorbing.'
    assert np.allclose(mdp.P[1][0], np.array([0.9, 0.1, 0])), 'taking the action LEFT from state 1 should go to state 0 with prob 0.9'
    assert np.allclose(mdp.P[2][1], np.array([0, 0, 1])), 'taking the action RIGHT from state 2 should go to state 2 with prob 1'
    assert np.allclose(mdp.P[2][0], np.array([0, 0.9, 0.1])), 'taking the action LEFT from state 2 should go to state 1 with prob 0.9'

    assert np.allclose(mdp.R[0][:], 0), 'No reward from terminal state'
    assert mdp.R[1][0] == +5, 'taking LEFT from state 1 should give +5 reward'
    assert mdp.R[1][1] == 0, 'taking RIGHT from state 1 should give 0 reward'
    assert np.allclose(mdp.R[2][:], 0), 'No reward from other states'
