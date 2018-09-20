import gym
import gym_windy

env = gym.make("wall-v0")
env.set_state_type('onehot')
env.reset()
while True:
    env.render()
    action = input("action:")
    print(action)
    print(env.step(action))