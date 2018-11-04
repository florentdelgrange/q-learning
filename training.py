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
import sys

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
    [0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.]  # Y + A + LEFT (accelerate LEFT and spin jump)

], dtype='?')  # meaningful actions for SuperMarioWorld

action_meaning = [
    "B button",
    "Y button",
    "UP button",
    "DOWN button",
    "LEFT button",
    "RIGHT button",
    "DOWN button + RIGHT",
    "DOWN button + LEFT",
    "A button",
    "Y + B + RIGHT",
    "Y + RIGHT",
    "Y + RIGHT + DOWN",
    "Y + A + RIGHT",
    "Y + B + LEFT",
    "Y + LEFT",
    "Y + LEFT + DOWN",
    "Y + A + LEFT"
]

align = lambda x: x if len(x) == 2 else x + " "
space = lambda x: " " * 8 if x >= 0 else " " * 7

global current_LOGS
current_LOGS = ""
ERASE_LINE = '\x1b[2K'
CURSOR_UP_ONE = '\x1b[1A'


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def custom_epsilon_greedy(strategy, epsilon, state):
    p = random.random()
    if p < epsilon:
        p = random.random()
        if p < 0.65 and epsilon >= 1:
            if p <= 0.65 / 2:
                return 10  # Y + RIGHT (accelerate RIGHT)
            else:
                return 9  # Y + B + RIGHT (accelerate RIGHT and jump)
        else:
            return random.randint(0, meaningful_actions.shape[0] - 1)
    else:
        q_values = strategy.get_q_values(state)
        best_action = np.argmax(q_values)
        # choose best actions w.r.t q-values
        best_actions = np.array(range(len(q_values)))[(q_values / q_values.max()) >= 0.7]
        # choose best q-values
        best_q_values = q_values[(q_values / q_values.max()) >= 0.7]
        proba = softmax(best_q_values)
        proba_on_q_values = np.zeros(shape=len(q_values))
        for i, a in enumerate(best_actions):
            proba_on_q_values[a] = best_q_values[i]

        logs = 4 * " " + "Q-values" + 8 * " " + "Softmax" + 7 * " " + "action meaning\n"
        logs += "\n".join(
            [align(str(i)) + ": " + "{:.6f}".format(x) + space(x) + "{:.4f}".format(proba_on_q_values[i]) + " " * 8 +
             action_meaning[i] for i, x in enumerate(q_values)]
        )
        #  softmax choice
        choice = np.random.choice(best_actions, 1, p=proba)[0]
        logs += "\nBest Action: {} | Action chosen: {}\n".format(best_action, choice)
        global current_LOGS
        current_LOGS = logs
        return choice


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


def liveness(reward):
    return reward if reward else -0.025


if __name__ == '__main__':

    args = docopt(__doc__)
    states_init = "             Forest1\
        Bridges2      \
        ChocolateIsland3  Forest5\
        DonutPlains1      Start             \
        DonutPlains2".split()
    states = "Bridges1          Forest2           YoshiIsland2\
        Bridges2          Forest3           YoshiIsland3\
        ChocolateIsland1  Forest4           YoshiIsland4\
        ChocolateIsland2  Forest5           \
        ChocolateIsland3  Start             \
        DonutPlains1      VanillaDome1      \
        DonutPlains2      VanillaDome2      \
        DonutPlains3      VanillaDome3      \
        DonutPlains4      VanillaDome4      \
        DonutPlains5      VanillaDome5      \
        Forest1           YoshiIsland1".split()
    t = 0
    epsilon = float(args['--epsilon'])
    strategy = None
    init_iterations = 20
    status = ""

    try:
        while True:
            if init_iterations:
                emulator_state = states_init[random.randint(0, len(states_init) - 1)]
            else:
                emulator_state = states[random.randint(0, len(states) - 1)]
            if t:
                env.load_state(emulator_state)
            else:
                env = retro.make("SuperMarioWorld-Snes", emulator_state, scenario='scenario2')
                #  show_all_actions_meaning(env)
            next_state = env.reset()
            input_shape = pre_process(next_state).shape
            t = 0
            total_reward = 0
            if not strategy:
                strategy = DQL(env, number_of_actions=len(meaningful_actions), input_shape=input_shape)
                custom_epsilon_greedy(strategy, 0, pre_process(next_state))
            else:
                strategy.environment = env
            while True:
                state = env.get_screen()
                state = pre_process(state)

                action = custom_epsilon_greedy(strategy, epsilon, state)

                next_state, reward, done, info = env.step(meaningful_actions[action])
                next_state = pre_process(next_state)
                strategy.update(state, action, liveness(reward), next_state, done)
                t += 1
                if t % 10 == 0:
                    env.render()
                    for _ in range(7 + len(meaningful_actions)):
                        sys.stdout.write(CURSOR_UP_ONE)
                        sys.stdout.write(ERASE_LINE)
                    strategy_logs = strategy.logs
                    status = strategy_logs if strategy_logs else status
                    sys.stdout.write("Emulator state: {}\n".format(emulator_state))
                    sys.stdout.write("Status: {}\n".format(status))
                    sys.stdout.write("ε={}\n".format(epsilon))
                    sys.stdout.write("current score={}\n".format(total_reward))
                    sys.stdout.write(current_LOGS)
                    sys.stdout.flush()
                if not t % 100:
                    plt.imsave("last_state.png", np.array(np.squeeze(state)))
                total_reward += liveness(reward)
                if done:
                    if not init_iterations:
                        epsilon *= 0.995  # decay epsilon at each episode
                    else:
                        init_iterations -= 1
                    env.render()
                    try:
                        status = "done! time=%i, reward=%d" % (t, total_reward)
                        print()
                    except EOFError:
                        exit(0)
                    break

    except KeyboardInterrupt:
        exit(0)
