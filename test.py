import gym
import gym_windy
from collections import deque

# env = gym.make("wall-v0")
# env.set_state_type('onehot')
# env.reset()
# while True:
#     env.render()
#     action = input("action:")
#     print(action)
#     print(env.step(action))


def one_hot(hot_index, arr_len):
    return [0 if i != hot_index else 1 for i in range(arr_len)]

