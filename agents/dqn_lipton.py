from collections import deque
import tensorflow as tf
from agents.dqn import DQNAgent
import copy
import sys
import numpy as np

class DQNLiptonAgent(DQNAgent):
    def __init__(self, num_actions, eps_min, eps_max, eps_decay_steps):
        # replay memory
        self.replay_memory_size = 20000
        self.replay_memory = deque([], maxlen=self.replay_memory_size)
        self.safe_memory = deque([], maxlen=self.replay_memory_size)
        self.danger_memory = deque([], maxlen=self.replay_memory_size)
        # el numero de pasos al estado peligroso
        self.nk = 2;
        # learning and environment
        self.num_actions = num_actions
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay_steps = eps_decay_steps
        # training
        self.batch_size = 50
        self.discount_rate = 0.99
        self.X_state = None
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

    def append_to_memory(self, state, action, reward, next_state, done):

        # esto solo es para debug
        # state = np.argmax(state, axis=0)
        # next_state = np.argmax(next_state, axis=0)

        current_tuple = (state, action, reward, next_state, 1.0 - done)
        self.replay_memory.append(current_tuple)

        # determinar si el estado es peligroso
        # esto se puede cambiar
        is_dangerous = reward < -1

        if is_dangerous:

            self.safe_memory = copy.deepcopy(self.replay_memory)

            for i in range(self.nk + 1):
                also_dangerous = self.safe_memory.pop()
                self.danger_memory.append(also_dangerous)
