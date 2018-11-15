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

# first_term = np.random.randint(0, 3, size=(1, 4))
# print("first_term", first_term.shape)
# print(first_term)
#
# second_term = np.random.randint(0, 2, size=(4, 5))
# print("second_term", second_term.shape)
# print(second_term)
#
# test = first_term.reshape(4, -1) - second_term
# print("test", test.shape)
# print(test)
#
# average = np.average(test, axis=1).reshape(-1, 4)
# print("average", average.shape)
# print(average)

test1 = np.random.randint(0, 100, size=(10, 1))
test2= np.random.randint(0, 100, size=(10, 1))
test3= np.random.randint(0, 100, size=(10, 1))
test4= np.random.randint(0, 100, size=(10, 1))
test = np.hstack((test1, test2))
test = np.hstack((test, test3))
test = np.hstack((test, test4))
print("1")
print(test1)
print("2")
print(test2)
print("3")
print(test3)
print("t")
print(test)
# print(np.argmax(test))