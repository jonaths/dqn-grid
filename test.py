import gym
import gym_windy
from collections import deque
import numpy as np

# env = gym.make("wall-v0")
# env.set_state_type('onehot')
# env.reset()
# while True:
#     env.render()
#     action = input("action:")
#     print(action)
#     print(env.step(action))

first_term = np.random.randint(0, 3, size=(4, 1))
print("first_term", first_term.shape)
print(first_term)

second_term = np.random.randint(0, 2, size=(1, 4, 3))
print("second_term", second_term.shape)
print(second_term)

test = np.expand_dims(first_term, axis=0) - second_term
print("test", test.shape)
print(test)

average = np.average(test, axis=2).reshape(4, -1)
print("average", average.shape)
print(average)
