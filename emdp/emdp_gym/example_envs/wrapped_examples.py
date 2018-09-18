"""Examples from emdp.examples wrapped as gym environments"""

from emdp import examples
from emdp.emdp_gym.gym_wrap import GymToMDP

class SB_example35(GymToMDP):
    def __init__(self):
        super().__init__(examples.build_SB_example35())

class twostate_MDP(GymToMDP):
    def __init__(self):
        super().__init__(examples.build_twostate_MDP())


