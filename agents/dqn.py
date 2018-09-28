import numpy as np
from collections import deque
from networks.qnetworks import conv_network, ff_network
import tensorflow as tf
from summaries.summaries import variable_summaries, simple_summaries


class DQNAgent(object):
    def __init__(self, X_state, num_actions, eps_min, eps_max, eps_decay_steps):
        self.X_state = X_state
        # replay memory
        self.replay_memory_size = 20000
        self.replay_memory = deque([], maxlen=self.replay_memory_size)
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
        # tensorflow saver and logger
        self.init = None
        self.merged = None
        self.saver = None

        pass

    def create_q_networks(self):

        self.online_q_values, self.online_vars = \
            ff_network(self.X_state, n_outputs=self.num_actions, name="q_networks/online")
        self.target_q_values, self.target_vars = \
            ff_network(self.X_state, n_outputs=self.num_actions, name="q_networks/target")

        # We need an operation to copy the online DQN to the target DQN
        self.copy_ops = [target_var.assign(self.online_vars[var_name])
                         for var_name, target_var in self.target_vars.items()]
        self.copy_online_to_target = tf.group(*self.copy_ops)

        # Now for the training operations
        with tf.variable_scope("q_train"):
            self.X_action = tf.placeholder(tf.int32, shape=[None])
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name="y_val")
            self.q_value = tf.reduce_sum(
                self.online_q_values * tf.one_hot(self.X_action, self.num_actions),
                axis=1, keep_dims=True)
            self.error = tf.abs(self.y - self.q_value)
            self.clipped_error = tf.clip_by_value(self.error, 0.0, 1.0)
            self.linear_error = 2 * (self.error - self.clipped_error)
            self.loss = tf.reduce_mean(tf.square(self.clipped_error) + self.linear_error)

            self.optimizer = tf.train.MomentumOptimizer(
                self.learning_rate, self.momentum, use_nesterov=True)
            self.training_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # agrupa los summaries en el grafo para que no aparezcan por todos lados
        with tf.name_scope('q_summaries'):
            simple_summaries(self.linear_error, 'linear_error')
            simple_summaries(self.loss, 'loss')
            simple_summaries(self.online_q_values, 'online_q_values')





    def append_to_memory(self, state, action, reward, next_state, done):
        # esto solo es para debug
        # state = np.argmax(state, axis=0)
        # next_state = np.argmax(next_state, axis=0)

        current_tuple = (state, action, reward, next_state, 1.0 - done)
        self.replay_memory.append(current_tuple)

    def sample_memories(self):
        indices = np.random.permutation(len(self.replay_memory))[:self.batch_size]
        cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
                cols[4].reshape(-1, 1))

    def epsilon_greedy(self, q_values, step):
        epsilon = max(self.eps_min,
                      self.eps_max - (self.eps_max - self.eps_min) * step / self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)  # random action
        else:
            return np.argmax(q_values)  # optimal action

    def preprocess_observation(self, obs):
        # img = obs[1:176:2, ::2]  # crop and downsize
        # img = img.mean(axis=2)  # to greyscale
        # img[img == mspacman_color] = 0  # Improve contrast
        # img = (img - 128) / 128 - 1  # normalize from -1. to 1.
        # return img.reshape(88, 80, 1)
        return obs
