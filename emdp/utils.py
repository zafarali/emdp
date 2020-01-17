import numpy as np

# 1D utilities.
def convert_int_rep_to_onehot(state, vector_size):
    s = np.zeros(vector_size)
    s[state] = 1
    return s

def convert_onehot_to_int(state):
    if type(state) is not np.ndarray:
        state = np.array(state)
    return state.argmax().item()
#
# def xy_to_flatten_state(state, size):
#     """Flatten state (x,y) into a one hot vector of size"""
#     idx = self.size * state[0] + state[1]
#     one_hot = np.zeros(self.size * self.size)
#     one_hot[idx] = 1
#     return one_hot
#
# def unflatten_state(self, onehot):
#     onehot = onehot.reshape(self.size, self.size)
#     x = onehot.argmax(0).max()
#     y = onehot.argmax(1).max()
#     return (x, y)

# def step(self, action):
#     """action must be the index of an action"""
#     # get the vector representing the next state probabilities:
#     current_state_idx = np.argmax(self.current_state)
#     next_state_probs = self.P[current_state_idx, action]
#     # sample the next state
#     sampled_next_state = np.random.choice(np.arange(self.P.shape[0]), p=next_state_probs)
#     # observe the reward
#     reward = self.r[current_state_idx, action]
#     self.current_state = self.convert_int_rep_to_onehot(sampled_next_state)
#     #         if reward > 0 :print(reward, current_state_idx, action)
#     return self.current_state, reward, sampled_next_state == self.P.shape[0] - 1, {}
