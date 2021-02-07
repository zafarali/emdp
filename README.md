# emdp

Easy MDPs implemented in a gym like interface with access to transition dynamics.

Jump to topics: [Installation](#installation) | [Grid World](#grid-world) | [Grid World->Plotting](#plotting-gridworlds) | [Grid World->Customize](#customization) | [Accessing transition dynamics](#accessing-transition-dynamics)

## Installation

`cd` into this directory and then run:

```bash
pip install -e .
```

## Usage

`emdp` can simulate arbitriary MDPs with and without absorbing states.

### Chain World

These are found in `emdp.chainworld`. A helper function is given to you to build worlds easily:

```python
from emdp.chainworld import build_chain_MDP
from emdp import actions
build_chain_MDP(n_states=7, p_success=0.9, reward_spec=[(5, actions.RIGHT, +1), (1, actions.LEFT, -1)]
                    starting_distribution=np.array([0,0,0,1,0,0,0]),
                    terminal_states=[0, 6], gamma=0.9)
```
This creates a 7 state MDP where the agent starts in the middle at the two ends are two terminal states.
Once the agent enters a terminal state it goes into the absorbing state. The agent executes the wrong action with prob 0.1
If the agent is at the left of the world and it takes an action LEFT it gets a -1 and goes into the abosrbing state.
otherwise it gets nothing. If the agent is at the right of the world and it takes an action RIGHT it gets a +1 otherwise it gets nothing.

### Grid World

Here we provide helper functions to create gridworlds and a simple function to build an empty gridworld.
```python
from emdp.gridworld import build_simple_grid
P = build_simple_grid(size=5, terminal_states=[(0, 4)], p_success=0.9)
```
Builds a simple 5x5 grid world where there is a terminal state at (0, 4). The probability of successfully executing the action is 0.9. This function returns the transition matrix.

For a full example, see how to build this example from the S&B book:

```python
import emdp.gridworld as gw

def build_SB_example35():
    """
    Example 3.5 from (Sutton and Barto, 2018) pg 60 (March 2018 version).
    A rectangular Gridworld representation of size 5 x 5.

    Quotation from book:
    At each state, four actions are possible: north, south, east, and west, which deterministically
    cause the agent to move one cell in the respective direction on the grid. Actions that
    would take the agent off the grid leave its location unchanged, but also result in a reward
    of −1. Other actions result in a reward of 0, except those that move the agent out of the
    special states A and B. From state A, all four actions yield a reward of +10 and take the
    agent to A'. From state B, all actions yield a reward of +5 and take the agent to B'
    """
    size = 5
    P = gw.build_simple_grid(size=size, p_success=1)
    # modify P to match dynamics from book.

    P[1, :, :] = 0 # first set the probability of all actions from state 1 to zero
    P[1, :, 21] = 1 # now set the probability of going from 1 to 21 with prob 1 for all actions

    P[3, :, :] = 0  # first set the probability of all actions from state 3 to zero
    P[3, :, 13] = 1  # now set the probability of going from 3 to 13 with prob 1 for all actions

    R = np.zeros((P.shape[0], P.shape[1])) # initialize a matrix of size |S|x|A|
    R[1, :] = +10
    R[3, :] = +1

    p0 = np.ones(P.shape[0])/P.shape[0] # uniform starting probability (assumed)
    gamma = 0.9

    terminal_states = []
    return gw.GridWorldMDP(P, R, gamma, p0, terminal_states, size)
```

To actually use this there is a gym like interface where you can move around:

```python

mdp = build_SB_example35()
state, reward, done, _ = mdp.step(actions.UP) # moves the agent up.
```

#### Plotting GridWorlds

There are some tools built in for quickly plotting trajectories obtained from the `GridWorldMDP`s.

```python
from emdp.gridworld import GridWorldPlotter
from emdp import actions
import random
gwp = GridWorldPlotter(mdp.size, mdp.has_absorbing_state) # alternatively you can use GridWorldPlotter.from_mdp(mdp)

# collect some trajectories from the GridWorldMDP object:

trajectories = []
for _ in range(3): # 3 trajectories
  trajectory = [mdp.reset()]
  for _ in range(10): # 10 steps maximum
    state, reward, done, info = mdp.step(random.sample([actions.LEFT, actions.RIGHT, 
                                                        actions.UP, actions.DOWN], 1)[0])
    trajectory.append(state)
  trajectories.append(trajectory)
```

Now `trajectories` contains a list of lists of numpy arrays which represent the states. You can easily obtain trajectory plots and state visitation heatmaps:

```python
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121)

# trajectory
gwp.plot_trajectories(ax, trajectories)
gwp.plot_grid(ax)

# heatmap
ax = fig.add_subplot(122)
gwp.plot_heatmap(ax, trajectories)
gwp.plot_grid(ax)
```

You will get something like this:

![image](https://user-images.githubusercontent.com/6295292/39479043-20587d32-4d32-11e8-82ae-7deddca8dc07.png)

#### Customization

There is an interface to add walls and blockages to the gridworld.

```python
from emdp.gridworld.builder_tools import TransitionMatrixBuilder
builder = TransitionMatrixBuilder(grid_size=5, has_terminal_state=False)
builder.add_grid([], p_success=1)
builder.add_wall_at((4, 2))
builder.add_wall_at((3, 2))
builder.add_wall_at((2, 2))
builder.add_wall_at((1, 2))
```

Using the above interface we have built a wall within our gridworld. Visualizing the trajectories in a similar way as above we get:

![image](https://user-images.githubusercontent.com/6295292/39713910-667750c2-51f6-11e8-9e80-b24b4861afda.png)


Alternatively, you can use `add_wall_between` which creates a straight line of walls between two positions on the grid.
So the following code will produce
```python
from emdp.gridworld.builder_tools import TransitionMatrixBuilder
builder = TransitionMatrixBuilder(grid_size=5, has_terminal_state=False)
builder.add_grid([], p_success=1)
builder.add_wall_between((0,2), (1, 2))
builder.add_wall_between((3,2), (4, 2))
builder.add_wall_between((1,1), (1, 3))
```

![image](https://user-images.githubusercontent.com/6295292/39715133-376e147e-51fa-11e8-98c4-d14528c330a6.png)


## Accessing transition dynamics

You can access transition dynamics by inspecting the `MDP` object:

```python
mdp.P # transition matrix
mdp.R # reward matrix
mdp.p0 # starting distribution
mdp.gamma # discount factor
mdp.terminal_states # the location of the terminal states
```

### Absorbing states

If you have an absorbing state in your MDP, it must be the last one. All actions executed in the absorbing state must lead to itself.


## Current usage

I use this code consistently in many pieces of work. If you use it, let me know and I'll be happy to add them here. Here are a few preprints that are available:


- [Bachman, Philip, et al. "Vfunc: a deep generative model for functions." arXiv preprint arXiv:1807.04106 (2018).](https://arxiv.org/abs/1807.04106)
- [Ahmed, Zafarali, et al. "Understanding the impact of entropy in policy learning." arXiv preprint arXiv:1811.11214 (2018).](https://arxiv.org/abs/1811.11214)


