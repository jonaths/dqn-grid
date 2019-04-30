import numpy as np
from plotters.plotter import PolicyPlotter
import matplotlib.pyplot as plt
import math
import sys


def is_top_or_bottom(number, num_rows=6):
    test = number % num_rows
    if test == 0:
        return True
    elif test == num_rows - 1:
        return True
    else:
        return False


def coord2ind(coord, num_rows, num_cols):
    """
    Converts coordinates to index
    :param num_cols:
    :param num_rows:
    :param coord: [x, y]
    :return:
    """

    [row, col] = coord

    assert (row < num_rows)
    assert (col < num_cols)

    return col * num_rows + row


def ind2coord(num_rows, index):
    """
    Converts an index to coordinates
    :param num_rows:
    :param index: int
    :return:
    """

    assert (index >= 0)
    # assert(index < self.n - 1)

    col = index // num_rows
    row = index % num_rows

    return [row, col]


def create_states_list_from_dict(states_dict, num_states):
    states_list = []
    for i in range(num_states):
        states_list.append(0)
        if i in states_dict.keys():
            states_list[i] = states_dict[i]
    return states_list


def process_obs(obs, params=None, name=None):
    if name == 'grid':
        return obs
    elif name == 'buckets':
        tup = discretize(obs, params)
        # esto debe ir de acuerdo al numero de buckets
        # el -2 es para considerar solo angulo y su derivada que son los dos ultimos
        int_state = coord2ind(tup[-2:], 6, 12)
        one_hot_state = np.zeros(params['num_states'])
        one_hot_state[int_state] = 1
        return one_hot_state
    else:
        raise Exception('Invalid name. ')


def discretize(obs, params):
    upper_bounds = [params['env'].observation_space.high[0], 0.5, params['env'].observation_space.high[2],
                    math.radians(50)]
    lower_bounds = [params['env'].observation_space.low[0], -0.5, params['env'].observation_space.low[2],
                    -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in
              range(len(obs))]
    new_obs = [int(round((params['buckets'][i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(params['buckets'][i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)


def save_risk_map(states_dict, num_states, num_rows, num_cols, name):
    # states_dict = {0: 5, 1: 2}
    states_list = np.array(create_states_list_from_dict(states_dict, num_states)).reshape(
        (num_states, 1, -1))

    plotter = PolicyPlotter(states_list, num_rows, num_cols)

    # tambien se puede generar el mapa de politicas
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    im, cbar = plotter.build_heatmap(annotate=False, cmap='Reds')
    fig.tight_layout()
    fig.savefig(name, dpi=100)


def save_policy(qtable, num_rows, num_cols, name, labels=['^', 'v']):

    plotter = PolicyPlotter(qtable, num_rows, num_cols)

    # tambien se puede generar el mapa de politicas
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    im, cbar, text = plotter.build_policy(
        labels=labels, show_numbers=False, cmap='Blues')
    fig.tight_layout()
    fig.savefig(name, dpi=100)
