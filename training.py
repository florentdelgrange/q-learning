"""Deep q-learning training for Super Mario World

Usage:
    training.py [--epsilon <epsilon>]
    training.py (-h | --help)

Options:
-h --help                           Display help.
-e --epsilon <epsilon>              Actions are selected randomly with a probability epsilon (epsilon-greedy algorithm). [default: 1.]
"""

import random

import cv2
import matplotlib
import numpy as np
import retro
from keras.utils import to_categorical
from docopt import docopt

from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from q_learning_strategies import DQL

meaningful_actions = np.array([
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # B button (normal jump)
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # Y button (accelerate -- same than X button -> we ignore it)
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # UP button
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # DOWN button
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # LEFT button
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # RIGHT button
    [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],  # DOWN button + RIGHT
    [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],  # DOWN button + LEFT
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # A button (spin jump)
    [1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # Y + B + RIGHT (accelerate RIGHT and jump)
    [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # Y + RIGHT (accelerate RIGHT)
    [0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],  # Y + RIGHT + DOWN (accelerate RIGHT and DOWN)
    [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],  # Y + A + RIGHT (accelerate RIGHT and spin jump)
    [1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # Y + B + LEFT (accelerate LEFT and jump)
    [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # Y + LEFT (accelerate LEFT)
    [0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],  # Y + LEFT (accelerate LEFT and DOWN)
    [0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.]   # Y + A + LEFT (accelerate LEFT and spin jump)

], dtype='?')  # meaningful actions for SuperMarioWorld


def custom_epsilon_greedy(strategy, epsilon, state, current_sum=0):
    p = random.random()
    if p < epsilon:
        p = random.random()
        if p < 0.65 * epsilon and current_sum > -200:
            if p <= 0.65 * epsilon / 2:
                return [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]  # Y + RIGHT (accelerate RIGHT)
            else:
                return [1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.] # Y + B + RIGHT (accelerate RIGHT and jump)
        else:
            return meaningful_actions[random.randint(0, meaningful_actions.shape[0] - 1)]
    else:
        return strategy.play(state)


def pre_process(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    # observation = observation[26:110, :]
    # observation = observation[:, :]
    # ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))


def show_all_actions_meaning(env):
    unique_actions = list(range(env.action_space.n))
    for action in to_categorical(unique_actions):
        print(action)
        print(env.get_action_meaning(action))
        print(" ")


def liveness(action, reward):
    return reward if (action[7] or reward) else -0.025


if __name__ == '__main__':

    args = docopt(__doc__)

    states = "             Forest1\
        Bridges2           YoshiIsland1\
                   YoshiIsland2\
        ChocolateIsland2  Forest4\
        ChocolateIsland3  Forest5\
        DonutPlains1      Start             \
        DonutPlains2      \
        DonutPlains3      \
        DonutPlains4      \
        VanillaDome4".split()
    t = 0
    epsilon = float(args['--epsilon'])
    strategy = None

    try:
        while True:
            state = states[random.randint(0, len(states) - 1)]
            print("State {} loaded.".format(state))
            if t:
                env.load_state(state)
            else:
                env = retro.make("SuperMarioWorld-Snes", state, scenario='scenario2')
                #  show_all_actions_meaning(env)
            next_state = env.reset()
            input_shape = pre_process(next_state).shape
            t = 0
            total_reward = 0
            if not strategy:
                # strategy = RandomStrategy(env)
                strategy = DQL(env, action_space=meaningful_actions, input_shape=input_shape)
            else:
                strategy.environment = env
            while True:
                state = env.get_screen()
                state = pre_process(state)

                action = custom_epsilon_greedy(strategy, epsilon, state, total_reward)

                next_state, reward, done, info = env.step(action)
                next_state = pre_process(next_state)
                strategy.update(state, action, liveness(action, reward), next_state, done)
                t += 1
                if t % 10 == 0:
                    env.render()
                if not t % 100:
                    plt.imsave("last_state.png", np.array(np.squeeze(state)))
                total_reward += liveness(action, reward)
                if reward > 0:
                    print('t=%i got reward: %g, total reward: %g' % (t, reward, total_reward))
                if reward < 0:
                    print('t=%i got penalty: %g, total reward: %g' % (t, -reward, total_reward))
                if done:
                    epsilon *= 0.996  # decay epsilon at each episode
                    print("epsilon={} -- probability to force RIGHT: {}".format(epsilon, 0.65 * epsilon**2))
                    env.render()
                    try:
                        print("done! time=%i, reward=%d" % (t, total_reward))
                        print()
                    except EOFError:
                        exit(0)
                    break

    except KeyboardInterrupt:
        exit(0)
