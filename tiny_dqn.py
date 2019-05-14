from __future__ import division, print_function, unicode_literals
import gym
import numpy as np
import os
import tensorflow as tf
from collections import namedtuple
from agents.dqn_lipton import DQNLiptonAgent
import sys
import time
from helpers import tools
import matplotlib.pyplot as plt


from plotters.line_plotter import LinesPlotter
from plotters.history import History

from helpers.policy_plotter import prepare_q_table, plot_policy, plot_heatmap

# crea la carpeta del experimento
exp_time = str(int(time.time()))
# exp_time = '1556490978'

path = os.getcwd()
exp_path = path + '/' + 'results/' + exp_time
os.mkdir(exp_path)
os.mkdir(exp_path + '/' + 'plots')

# al parecer no aprende. la grafica de recompen sa no convrge

args_struct = namedtuple(
    'args',
    'env_name buckets number_steps learn_iterations, save_steps copy_steps '
    'render path test load verbosity training_start batch_size ')
args = args_struct(
    env_name='CartPole-v0',
    buckets=(1, 1, 6, 12),
    number_steps=100000,
    learn_iterations=4,
    training_start=1000,
    save_steps=1000,
    copy_steps=500,

    render=False,
    # render=True,

    path='results/' + exp_time + '/' + 'models/my_dqn.ckpt',

    test=False,
    # test=True,

    load=False,
    # load='results/' + '1556820556' + '/' + 'models/my_dqn.ckpt',

    verbosity=1,
    batch_size=90
)

print("Args:")
print(args)

env = gym.make('CartPole-v0')

# atari ----------------------
# input_height = 88
# input_width = 80
# input_channels = 1
# ----------------------------

input_height = 6
input_width = 12
input_channels = 1

process_params = {
    'env': env,
    'buckets': args.buckets,
    'num_states': input_height * input_width
}

# danger_states = [1, 70]
danger_states = [0, 1, 2, 3, 4, 5, 66, 67, 68, 69, 70, 71]
#
# aqui voy... juntar en un data.npy los ultimos 3 experimentos y graficar
# luego ver si puedo mejorar la grafica de aprendizaje y fallo variando parametros de la red

n_outputs = env.action_space.n

# X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
X_state = tf.placeholder(tf.float32, shape=[None, input_height * input_width])

# And on to the epsilon-greedy policy with decaying epsilon
eps_min = 0.05
eps_max = 1.0 if not args.test else eps_min
eps_decay_steps = int(args.number_steps * 0.50)

# We need to preprocess the images to speed up training
mspacman_color = np.array([210, 164, 74]).mean()

agent = DQNLiptonAgent(X_state, n_outputs, eps_min, eps_max, eps_decay_steps)
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

# We will keep track of the max Q-Value over time and compute the mean per game
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0

plotter = LinesPlotter(['reward', 'steps', 'end_state'], 1, 3000)
history = History()
episode_count = 0

# game over, start again
obs = env.reset()
state = tools.process_obs(obs,
                          name='buckets',
                          params=process_params)
done = False

with tf.Session() as sess:
    if args.load and os.path.isfile(args.load + ".index"):
        agent.saver.restore(sess, args.load)
    else:
        agent.init.run()
        agent.copy_online_to_target.run()

    writer = tf.summary.FileWriter('results/' + exp_time + '/' + "outputs", sess.graph)

    while True:

        step = agent.global_step.eval()

        if step >= args.number_steps:
            break

        iteration += 1

        if args.verbosity > 0:
            print('\rIteration {}   Training step {}/{} ({:.1f})%   '
                  'Loss {:5f}    Mean Max-Q {:5f}   Episode {}   '.format(
                iteration, step, args.number_steps, step * 100 / args.number_steps,
                loss_val, mean_max_q, episode_count), end="")

        if done:
            # print(history.get_state_sequence())
            # print(history.get_steps_count(), history.get_state_sequence()[-1])
            # input()
            plotter.add_episode_to_experiment(0, episode_count,
                                              [
                                                  history.get_total_reward(),
                                                  history.get_steps_count(),
                                                  history.get_state_sequence()[-1]
                                              ])
            history.clear()
            episode_count += 1

            # game over, start again
            obs = env.reset()
            state = tools.process_obs(obs,
                                      name='buckets',
                                      params=process_params)

        # aqui voy... necesito:
        #     checar la lngitud de los episodios para saber
        #     que longitud dar a k
        #     graficar la recompensa en 1556493818

        if args.render:
            env.render()

        if episode_count in [1000, 3000, 5000, 7000]:

            # if args.save_policy:
            step_prefix = '{:05d}'.format(episode_count) + '-'
            # las etiquetas en el orden de la tabla q
            labels = ['^', 'v']

            # genera el espacio de estados y recupera los valores q
            q_table = prepare_q_table(input_height, input_width, n_outputs, agent)

            # genera grafica de politica
            plot_policy(
                q_table, input_height, input_width, labels,
                file_name='results/' + exp_time + '/' + 'plots/' + step_prefix + 'policy.png')

            risk_map = []
            num_states = input_height * input_width
            for s in range(num_states):
                state_one_hot = np.eye(num_states)[s]
                fear = agent.online_fear.eval(feed_dict={X_state: [state_one_hot]})
                risk_map.append(fear)
            # el 1 - es para que lo oscuro sea lo seguro
            plot_heatmap(1-np.array(risk_map), input_height, input_width, index=None,
                         file_name='results/' + exp_time + '/' + 'plots/' + step_prefix + 'riskmap.png')

        # Online DQN evaluates what to do
        q_values = agent.online_q_values.eval(feed_dict={X_state: [state]})
        action = agent.epsilon_greedy(q_values, step)

        # print(q_values)
        # print(action)
        # input('')

        # action = input("Action: ")
        # fear = agent.online_fear.eval(feed_dict={X_state: [state]})
        # print("fear", fear)

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = tools.process_obs(obs,
                                       name='buckets',
                                       params=process_params)

        # se asegura que los estados sean enteros
        state_ind = np.argmax(state)
        next_state_ind = np.argmax(next_state)
        history.insert((state_ind, action, reward, next_state_ind))

        is_dangerous = True if next_state_ind in danger_states else False

        # Let's memorize what happened
        agent.append_to_memory(state, action, reward, next_state, done, is_dangerous)
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
        fear_val = (agent.sample_memories())

        next_q_values = agent.target_q_values.eval(feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)

        # normal dqn
        # y_val = rewards + continues * agent.discount_rate * max_next_q_values
        # lipton dqn
        # y_val = rewards \
        #         + continues * agent.discount_rate * max_next_q_values \
        #         - agent.get_lambda(step) * 1

        fear = agent.online_fear.eval(feed_dict={X_state: [state]})
        risk_penalization = abs(agent.get_lambda(step) * fear)
        y_val = rewards + continues * agent.discount_rate * max_next_q_values - risk_penalization



        # print("XXX")
        # print(y_val.shape)
        # print(agent.y)
        # print(fear_val.shape)
        # print(agent.fear_val)

        # Train the online DQN

        # aqui voy... comente la creacion de la red fear y elimine su entrenamiento
        # al incluirlo no aprendio
        # probar incluyendo la red sin su entrenamiento, luego el entrenamiento nuevamente

        # entrenar red q
        _, loss_val = sess.run(
            [
                agent.training_op, agent.loss
            ],
            feed_dict={
                agent.X_state: X_state_val,
                agent.X_action: X_action_val,
                agent.y: y_val,
                agent.fear_val: fear_val
            })

        # entrenar red fear
        _, fear_loss_val = sess.run(
            [
                agent.fear_training_op, agent.fear_loss,
            ],
            feed_dict={
                agent.X_state: X_state_val,
                agent.X_action: X_action_val,
                agent.y: y_val,
                agent.fear_val: fear_val
            })

        # correr summaries
        summary, = sess.run(
            [
                agent.merged
            ],
            feed_dict={
                agent.X_state: X_state_val,
                agent.X_action: X_action_val,
                agent.y: y_val,
                agent.fear_val: fear_val
            })

        if step % 50 == 0:
            writer.add_summary(summary, step)

        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            agent.copy_online_to_target.run()

        # And save regularly
        if step % args.save_steps == 0:
            agent.saver.save(sess, args.path)

    plotter.save_data('results/' + exp_time + '/' + 'data')


