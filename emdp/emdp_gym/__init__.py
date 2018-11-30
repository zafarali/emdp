from emdp.emdp_gym.gym_wrap import gymify
from gym.envs.registration import register

register(
    id='sb-example3-v0',
    entry_point='emdp.emdp_gym.example_envs:SB_example35',
)
register(
    id='two-state-v0',
    entry_point='emdp.emdp_gym.example_envs:twostate_MDP',
)