from __future__ import division, print_function, unicode_literals
from collections import deque
import gym
import gym_windy
import numpy as np
import os
import tensorflow as tf
from networks.qnetworks import conv_network, ff_network
from summaries.summaries import variable_summaries, simple_summaries
from collections import namedtuple
from agents.dqn_dist_lipton_v4 import DQNDistributiveLiptonAgent
import sys
import time
from  helpers.calc_fear_penalized import calc_fear_value

args_struct = namedtuple(
    'args',
    'number_steps learn_iterations, save_steps copy_steps '
    'render path test verbosity training_start batch_size ')
args = args_struct(
    number_steps=50000,
    learn_iterations=4,
    training_start=1000,
    save_steps=1000,
    copy_steps=500,
    # render=False,
    render=True,
    path='models/my_dqn.ckpt',
    # test=False,
    test=True,
    verbosity=1,
    batch_size=90
)

print("Args:")
print(args)

# atari -----------------------
# env = gym.make("MsPacman-v0")
# -----------------------------

env = gym.make("border-v0")
env.set_state_type('onehot')

done = True  # env needs to be reset

# atari ----------------------
# input_height = 88
# input_width = 80
# input_channels = 1
# ----------------------------

input_height = env.rows
input_width = env.cols
input_channels = 1

n_outputs = env.action_space.n  # 9 discrete actions are available

# el dqn original
# X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])

# la idea distributiva donde la fear network solo depende del estado
# X_state = tf.placeholder(tf.float32, shape=[None, input_height * input_width])


X_state = tf.placeholder(tf.float32, shape=[None, input_height * input_width])
X_state_action = tf.placeholder(tf.float32, shape=[None, input_height * input_width + n_outputs])

# And on to the epsilon-greedy policy with decaying epsilon
eps_min = 0.1
eps_max = 1.0 if not args.test else eps_min
eps_decay_steps = args.number_steps // 2

# We need to preprocess the images to speed up training
mspacman_color = np.array([210, 164, 74]).mean()

agent = DQNDistributiveLiptonAgent(X_state, X_state_action, n_outputs, eps_min, eps_max,
                                   eps_decay_steps)
agent.create_q_networks()
agent.create_fear_networks()
agent.init = tf.global_variables_initializer()
agent.saver = tf.train.Saver()

# evita agregar al grafo los summaries uno por uno
agent.merged = tf.summary.merge_all()

# TensorFlow - Execution phase
training_start = args.training_start
skip_start = 0  # Skip the start of every game (it's just waiting time).
# skip_start = 90  # Skip the start of every game (it's just waiting time).
iteration = 0  # game iterations
done = True  # env needs to be reset

# We will keep track of the max Q-Value over time and compute the mean per game
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0

with tf.Session() as sess:
    if os.path.isfile(args.path + ".index"):
        agent.saver.restore(sess, args.path)
    else:
        agent.init.run()
        agent.copy_online_to_target.run()

    log_file = 'outputs/' + str(int(time.time())) + "_lmb-0.00_n-8"

    writer = tf.summary.FileWriter(log_file, sess.graph)

    while True:
        step = agent.global_step.eval()
        if step >= args.number_steps:
            break
        iteration += 1
        if args.verbosity > 0:
            print("\rIteration {}   Training step {}/{} ({:.1f})%   "
                  "Loss {:5f}    Mean Max-Q {:5f}   ".format(
                iteration, step, args.number_steps, step * 100 / args.number_steps,
                loss_val, mean_max_q), end="")
        if done:
            # game over, start again
            obs = env.reset()
            for skip in range(skip_start):  # skip the start of each game
                obs, reward, done, info = env.step(0)
            state = agent.preprocess_observation(obs)

        if args.render:
            env.render()

        curr_lmb = agent.get_lambda()

        # Online DQN evaluates what to do
        q_values = agent.online_q_values.eval(feed_dict={agent.X_state: [state]})
        action = agent.epsilon_greedy(q_values, step)

        # fear = agent.get_state_actions(state)
        # print()
        # print("lambda", curr_lmb, step)
        # print("state", np.argmax(state))
        # print("fear\n", fear)
        # print("q_values \n ", agent.get_online_q_values(state, 'normal'))
        # print("q_values' \n ", agent.get_online_q_values(state, 2))
        # print("action", action)
        # action = input('action: ')

        # aqui voy... al parecer aprende bien el fear model con action y state
        # ahora falta restarlo a la q y ver si aprende completo

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = agent.preprocess_observation(obs)

        # Let's memorize what happened
        agent.append_to_memory(state, action, reward, next_state, done)
        state = next_state

        if args.test:
            continue

        # Compute statistics for tracking progress (not shown in the book)
        total_max_q += q_values.max()
        game_length += 1
        if done:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        if iteration < training_start or iteration % args.learn_iterations != 0:
            # only train after warmup period and at regular intervals
            continue

        # Genera una tupla con batch_size muestras de cada parametro del MDP
        # mas una etiqueta que 1 o 0 que indica si la muestra es de la lista
        # danger o safe
        X_state_val, \
        X_action_val, \
        rewards, \
        X_next_state_val, \
        continues, \
        fear_labels, \
        X_state_action_val,\
        debug = (agent.sample_memories)

        # index = np.random.randint(len(continues) - 1)
        # print(debug)
        # print(fear_labels)
        # sys.exit(0)

        next_q_values = agent.target_q_values.eval(feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)

        # fear = agent.online_fear_softmax.eval(feed_dict={X_state: [state]})
        # action_state = np.append(agent.one_hot(action, n_outputs), state)
        # fear = agent.online_fear_softmax.eval(feed_dict={X_state_action: [action_state]})

        # normal dqn
        # y_val = rewards + continues * agent.discount_rate * max_next_q_values

        # lipton dqn
        # y_val = rewards + \
        #         continues * agent.discount_rate * max_next_q_values - \
        #         agent.get_lambda(step) * fear

        # lipton distributive dqn (only distributive fear)
        # y_val = rewards + \
        #         continues * agent.discount_rate * max_next_q_values - \
        #         curr_lmb * fear[:, 0].reshape(-1, 1)

        # la idea para restar que al parecer no funciona
        # action_state = np.append(agent.one_hot(action, n_outputs), state)
        # fear = agent.online_fear_softmax.eval(feed_dict={X_state_action: [action_state]})
        # y_val = np.average(rewards +
        #                    continues * agent.discount_rate * max_next_q_values -
        #                    agent.get_lambda() * fear, axis=1).reshape(-1, 1)

        # nueva idea con fear descontado
        fear_probs = agent.online_fear_softmax.eval(feed_dict={X_state_action: X_state_action_val})
        y_val = rewards + continues * agent.discount_rate * max_next_q_values \
                - calc_fear_value(fear_probs, lmb=agent.get_lambda()).reshape(-1, 1)

        # Train the online DQN

        # entrenar red q
        _, loss_val = sess.run(
            [
                agent.training_op, agent.loss
            ],
            feed_dict={
                agent.X_state: X_state_val,
                agent.X_action: X_action_val,
                agent.y: y_val,
                agent.fear_val: fear_labels
            })

        # entrenar red fear
        _, fear_loss_val = sess.run(
            [
                agent.fear_training_op, agent.fear_cost,
            ],
            feed_dict={
                agent.X_state: X_next_state_val,
                agent.X_state_action: X_state_action_val,
                agent.X_action: X_action_val,
                agent.y: y_val,
                agent.fear_val: fear_labels
            })

        # correr summaries
        summary, = sess.run(
            [
                agent.merged
            ],
            feed_dict={
                agent.X_state: X_state_val,
                agent.X_state_action: X_state_action_val,
                agent.X_action: X_action_val,
                agent.y: y_val,
                agent.fear_val: fear_labels
            })

        if step % 50 == 0:
            writer.add_summary(summary, step)

        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            agent.copy_online_to_target.run()

        # And save regularly
        if step % args.save_steps == 0:
            agent.saver.save(sess, args.path)
