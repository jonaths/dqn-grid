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

print("first_term")
first_term = np.random.randint(0, 3, size=(5, 1))
print(first_term)

print("second_term")
second_term = np.random.randint(0, 2, size=(1, 4))
print(second_term)

print("test")
test = first_term - second_term
print(test)

print("average")
average = np.average(test, axis=1)
print(average)
