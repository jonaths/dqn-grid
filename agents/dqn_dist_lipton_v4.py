from collections import deque
import tensorflow as tf
from agents.dqn import DQNAgent
from networks.qnetworks import ff_network
from summaries.summaries import simple_summaries
import copy
import sys
import numpy as np


class DQNDistributiveLiptonAgent(DQNAgent):
    def __init__(self, X_state, X_state_action, num_actions, eps_min, eps_max, eps_decay_steps):
        self.X_state = X_state
        self.X_state_action = X_state_action
        self.replay_memory_size = 20000
        self.replay_memory = deque([], maxlen=self.replay_memory_size)
        self.temp_replay_memory = deque([], maxlen=self.replay_memory_size)
        self.safe_memory = deque([], maxlen=self.replay_memory_size)
        self.danger_memory = deque([], maxlen=self.replay_memory_size)
        # el numero de pasos al estado peligroso
        self.nk = 2
        self.k_bins = 4
        self.k_steps = 1
        self.k_dict = {}
        for b in range(self.k_bins + 1):
            self.k_dict[b] = deque([], maxlen=50)
        # learning and environment
        self.num_actions = num_actions
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay_steps = eps_decay_steps
        # training
        self.batch_size = 50
        self.discount_rate = 0.99
        self.online_q_values = None
        self.online_vars = None
        self.target_q_values = None
        self.target_vars = None
        self.copy_ops = None
        self.copy_online_to_target = None
        self.learning_rate = 0.01
        self.momentum = 0.95
        self.X_action = None
        self.y = None
        self.q_value = None
        self.error = None
        self.clipped_error = None
        self.linear_error = None
        self.loss = None
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.optimizer = None
        self.training_op = None
        # fear
        self.fear_val = None
        self.online_fear = None
        self.online_fear_softmax = None
        self.fear_cross_entropy = None
        self.fear_cost = None
        self.fear_optimizer = None
        self.fear_training_op = None
        self.penalized_q = None
        # tensorflow saver and logger
        self.init = None
        self.merged = None
        self.saver = None
        self.lmb = 9.0
        self.lmb_phase_in = 50000

        pass

    def get_lambda(self):
        steps = self.global_step.eval()
        lmb = min(self.lmb, 1. * self.lmb * steps / self.lmb_phase_in)
        return lmb
        # return 0.

    def create_fear_networks(self):
        pass

        self.online_fear, _ = \
            ff_network(self.X_state_action, n_outputs=self.k_bins + 1, name="fear_networks/online")

        # Now for the training operations
        with tf.variable_scope("fear_train"):
            # self.X_state = tf.placeholder(tf.int32, shape=[None])
            self.fear_val = tf.placeholder(tf.float32, shape=[None, self.k_bins + 1], name="fear_val")

            self.fear_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.online_fear,
                labels=self.fear_val)
            self.fear_cost = tf.reduce_mean(self.fear_cross_entropy)

            self.fear_optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.fear_training_op = \
                self.fear_optimizer.minimize(self.fear_cost, global_step=self.global_step)

        self.online_fear_softmax = tf.nn.softmax(self.online_fear)

        # agrupa los summaries en el grafo para que no aparezcan por todos lados
        with tf.name_scope('fear_summaries'):
            simple_summaries(self.fear_cost, 'cost')

    def append_to_memory(self, state, action, reward, next_state, done):

        # esto solo es para debug
        # state = np.argmax(state, axis=0)
        # next_state = np.argmax(next_state, axis=0)

        current_tuple = (state, action, reward, next_state, 1.0 - done)
        self.replay_memory.append(current_tuple)
        self.temp_replay_memory.append(current_tuple)

        # determinar si el estado es peligroso
        # esto se puede cambiar
        is_dangerous = reward < -1

        if is_dangerous:
            # el primer elemento es peligroso y va en el indice 0
            for b in range(self.k_bins):
                if b == 0:
                    self.k_dict[b].append(self.temp_replay_memory.pop())
                    continue
                try:
                    # si el deque se queda sin elementos simplemente sigue
                    for s in range(self.k_steps):
                        # guarda los bins
                        self.k_dict[b].append(self.temp_replay_memory.pop())
                except IndexError:
                    continue
            # lo que quede en el deque va al ultimo indice del dict
            self.k_dict[b + 1].extend(self.temp_replay_memory)
            self.temp_replay_memory.clear()

    @staticmethod
    def one_hot(hot_index, arr_len):
        return [0 if i != hot_index else 1 for i in range(arr_len)]

    @property
    def sample_memories(self):

        # aqui voy... ahora hacer el sampling desde self.k_dict

        # la muestra para entrenamiento
        sample = []

        for k in self.k_dict:
            slice_size = self.batch_size / self.k_bins
            indices = np.random.permutation(len(self.k_dict[k]))[:slice_size]
            for i in indices:
                # agrega la muestra de la memoria segura y una etiqueta de 0
                current_sample = self.k_dict[k][i]
                # la tupla es <s, a, s'>, si lo pongo en k=0 quiere decir que s' es peligroso
                # le sumo 1 a la k actual porque si k' es peligroso quiere decir que s esta a k + 1
                # pasos del estado peligroso
                # lo que quiero es que le meta s y a al modelo y me de una distribucion de s'
                k_bins_one_hot = \
                    self.one_hot(k + 1 if k < self.k_bins else self.k_bins, self.k_bins + 1)
                action_one_hot = self.one_hot(self.k_dict[k][i][1], self.num_actions)
                new_sample = current_sample \
                             + (k_bins_one_hot,) \
                             + (np.append(action_one_hot, current_sample[0]),)
                sample.append(new_sample)

        # state, action, reward, next_state, continue, fear_prob (one_hot), action + state (one hot)
        cols = [[], [], [], [], [], [], []]
        for s in sample:
            for col, value in zip(cols, s):
                col.append(value)

        cols = [np.array(col) for col in cols]

        debug = np.hstack((np.argmax(cols[0], axis=1).reshape(-1, 1), cols[1].reshape(-1, 1)))
        debug = np.hstack((debug, np.argmax(cols[3], axis=1).reshape(-1, 1)))
        debug = np.hstack((debug, np.argmax(cols[5], axis=1).reshape(-1, 1)))

        # una tupla con batch_sizes muestras de cada campo
        return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
                cols[4].reshape(-1, 1), cols[5], cols[6], debug
                )

        # # s como entero
        # np.argmax(cols[0], axis=1).reshape(-1, 1),
        # # la accion como entero
        # cols[1].reshape(-1, 1),
        # # s' como entero
        # np.argmax(cols[3], axis=1).reshape(-1, 1),

    def get_state_actions(self, state):
        state_actions = \
            [np.append(self.one_hot(a, self.num_actions), state) for a in range(self.num_actions)]
        fear_array = np.array(
            self.online_fear_softmax.eval(feed_dict={self.X_state_action: state_actions})
        )
        return fear_array

    def get_online_q_values(self, state, mode='normal'):
        """
        Usa el modelo aprendido y recupera los valores q para un estado
        :param mode:
        :param state:
        :return:
        """
        original_q_values = self.online_q_values.eval(feed_dict={self.X_state: [state]})
        if mode == 'normal':
            return original_q_values
        else:
            fear = np.array(self.get_state_actions(state))
            cummulative_fear = np.cumsum(fear, axis=1)
            # print("cummulative_fear")
            # print(cummulative_fear)
            # print("original_q_values")
            # print(original_q_values)
            penalized_q = np.array(
                original_q_values.reshape(self.num_actions, -1) - self.get_lambda() * cummulative_fear)
            # print("penalized_q")
            # print(penalized_q)
            if mode == 'average':
                return np.average(penalized_q, axis=1).reshape(-1, self.num_actions)
            elif type(mode) == int:
                return penalized_q[:, mode]
            else:
                raise Exception('Not Implemented')
