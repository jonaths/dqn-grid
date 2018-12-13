import numpy as np


def calc_fear_penalized_q(q_values, fear_prob, gamma=0.9, lmb=1., k_bins=5, k_steps=1):
    """
    Calcula una penalizacion en funcion de la probabilidad de fallo.
    :param q_values: (?, num_actions)
    :param fear_prob: (?, num_actions, num_bins)
    :param gamma: tasa de descuento
    :param lmb: penalizacion por fallo
    :param k_bins: numero de pasos al fallo
    :param k_steps: numero pasos por cada bin
    :return: (?, num_actions). La penalizacion a q_values considerando la probabilidad de que
    falle en determinado de pasos descontado por gamma.
    """
    # crea los exponentes k(i-1)+1
    exp = k_steps * (np.array(range(k_bins)) + 1 - 1) + 1
    # crea la base (vector de gammas)
    base = np.ones(k_bins, ) * gamma
    # crea factores de descuento
    gamma_exp = np.power(base, exp)
    # calcula la penalizacion para cada accion considerando lambda
    fear_val_sum = np.sum(lmb * fear_prob * gamma_exp, axis=1)
    return q_values - fear_val_sum
