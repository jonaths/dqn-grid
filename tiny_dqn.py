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

args_struct = namedtuple(
    'args',
    'number_steps learn_iterations, save_steps copy_steps '
    'render path test verbosity training_start batch_size ')
args = args_struct(
    number_steps=50000,
    learn_iterations=4,
    training_start=1000,
    save_steps=1000,
    copy_steps=1000,
    render=False,
    # render=True,
    path='model/my_dqn.ckpt',
    test=False,
    # test=True,
    verbosity=1,
    batch_size=90
)

print("Args:")
print(args)

# atari -----------------------
# env = gym.make("MsPacman-v0")
# -----------------------------

env = gym.make("wall-v0")
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

# X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
X_state = tf.placeholder(tf.float32, shape=[None, input_height * input_width])

online_q_values, online_vars = ff_network(X_state, n_outputs=n_outputs, name="q_networks/online")
target_q_values, target_vars = ff_network(X_state, n_outputs=n_outputs, name="q_networks/target")

# We need an operation to copy the online DQN to the target DQN
copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

# Now for the training operations
learning_rate = 0.05
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keep_dims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

# agrupa los summaries en el grafo para que no aparezcan por todos lados
with tf.name_scope('summaries'):
    simple_summaries(linear_error, 'linear_error')
    simple_summaries(loss, 'loss')
    simple_summaries(online_q_values, 'online_q_values')


# evita agregar al grafo los summaries uno por uno
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Let's implement a simple replay memory
replay_memory_size = 20000
replay_memory = deque([], maxlen=replay_memory_size)


def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
            cols[4].reshape(-1, 1))


# And on to the epsilon-greedy policy with decaying epsilon
eps_min = 0.1
eps_max = 1.0 if not args.test else eps_min
eps_decay_steps = args.number_steps // 2


def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)  # random action
    else:
        return np.argmax(q_values)  # optimal action


# We need to preprocess the images to speed up training
mspacman_color = np.array([210, 164, 74]).mean()


def preprocess_observation(obs):
    # img = obs[1:176:2, ::2]  # crop and downsize
    # img = img.mean(axis=2)  # to greyscale
    # img[img == mspacman_color] = 0  # Improve contrast
    # img = (img - 128) / 128 - 1  # normalize from -1. to 1.
    # return img.reshape(88, 80, 1)
    return obs


# TensorFlow - Execution phase
training_start = args.training_start
discount_rate = 0.99
skip_start = 0  # Skip the start of every game (it's just waiting time).
# skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
done = True  # env needs to be reset

# We will keep track of the max Q-Value over time and compute the mean per game
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0

with tf.Session() as sess:
    if os.path.isfile(args.path + ".index"):
        saver.restore(sess, args.path)
    else:
        init.run()
        copy_online_to_target.run()

    writer = tf.summary.FileWriter("output", sess.graph)

    while True:
        step = global_step.eval()
        if step >= args.number_steps:
            break
        iteration += 1
        if args.verbosity > 0:
            print("\rIteration {}   Training step {}/{} ({:.1f})%   "
                  "Loss {:5f}    Mean Max-Q {:5f}   ".format(
                iteration, step, args.number_steps, step * 100 / args.number_steps,
                loss_val, mean_max_q), end="")
        if done:  # game over, start again
            obs = env.reset()
            for skip in range(skip_start):  # skip the start of each game
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)

        if args.render:
            env.render()

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        # Let's memorize what happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
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
            continue  # only train after warmup period and at regular intervals

        # Sample memories and use the target DQN to produce the target Q-Value
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
        next_q_values = target_q_values.eval(feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN
        _, loss_val, summary = sess.run([training_op, loss, merged],
                               feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})

        if step % 10 == 0:
            writer.add_summary(summary, step)

        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            copy_online_to_target.run()

        # And save regularly
        if step % args.save_steps == 0:
            saver.save(sess, args.path)
