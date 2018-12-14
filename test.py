import gym
import gym_windy
from collections import deque
import numpy as np
from helpers.calc_fear_penalized import calc_fear_value


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

# env = gym.make("wall-v0")
# env.set_state_type('onehot')
# env.reset()
# while True:
#     env.render()
#     action = input("action:")
#     print(action)
#     print(env.step(action))

q_values = np.random.randint(0, 3, size=(4, ))
print("q_values")
print(q_values)

fear_prob = np.array([
    [0., 0., 0., 0., 1.],
    [1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1.],
    [1., 0., 0., 0., 0.]])
print("fear")
print(fear_prob)

gamma = 0.9
lmb = 1.
k_bins = 5
k_steps = 1

exp = np.array(range(k_bins)) + 1
print(exp)

exp = k_steps * (exp - 1) + 1
print("exp")
print(exp)

base = np.ones(k_bins, ) * gamma
print("base")
print(base)

gamma_exp = np.power(base, exp)
print("gamma_exp")
print(gamma_exp)

fear_val = lmb * fear_prob * gamma_exp
print("fear_val")
print(fear_val)

fear_val_sum = np.sum(fear_val, axis=1)
print("fear_val_sum")
print(fear_val_sum)

penalized_q = q_values - fear_val_sum
print(penalized_q)

print("XXX")
print(q_values - calc_fear_value(fear_prob))

# print(np.sum(res))
#
# fear_value = res * f
# print("fear_value")
# print(fear_value)
