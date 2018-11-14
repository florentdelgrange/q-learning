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

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from q_learning_strategies import DQLStrategy

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

action_meaning = None

align = lambda x: x if len(x) == 2 else x + " "
space = lambda x: " " * 8 if x >= 0 else " " * 7

global current_LOGS
current_LOGS = ""
ERASE_LINE = '\x1b[2K'
CURSOR_UP_ONE = '\x1b[1A'


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def custom_epsilon_greedy(strategy, epsilon, state, max_ratio=0.55):
    p = random.random()
    if p < epsilon:
        p = random.random()
        if p < 0.6 * epsilon and epsilon >= 0.9:
            if p <= 0.6 * epsilon / 2:
                return 10  # Y + RIGHT (accelerate RIGHT)
            else:
                return 9  # Y + B + RIGHT (accelerate RIGHT and jump)
        else:
            return random.randint(0, meaningful_actions.shape[0] - 1)
    else:
        q_values = strategy.get_q_values(state)
        best_action = np.argmax(q_values)
        if not q_values.max():
            proba_on_q_values = np.ones(len(q_values)) / len(q_values)
        else:
            # choose best actions w.r.t q-values
            best_actions = np.array(range(len(q_values)))[(q_values / q_values.max()) >= max_ratio]
            # choose best q-values
            best_q_values = q_values[(q_values / q_values.max()) >= max_ratio]
            proba = softmax(best_q_values)
            proba_on_q_values = np.zeros(shape=len(q_values))
            for i, a in enumerate(best_actions):
                proba_on_q_values[a] = proba[i]

        logs = 4 * " " + "Q-values" + 8 * " " + "Softmax" + 7 * " " + "action meaning\n"
        logs += "\n".join(
            [align(str(i)) + ": " + "{:.6f}".format(x) + space(x) + "{:.4f}".format(proba_on_q_values[i]) + " " * 8 +
             action_meaning[i] for i, x in enumerate(q_values)]
        )
        #  softmax choice
        global current_LOGS
        if epsilon > 0.5 or not q_values.max():
            logs += "\nBest Action: {}\n".format(best_action)
            current_LOGS = logs
            return best_action
        else:
            choice = np.random.choice(best_actions, 1, p=proba)[0]
            logs += "\nBest Action: {} | Action chosen: {}\n".format(best_action, choice)
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


def liveness(reward, action):
    return reward if reward or action[7] else -0.0125


if __name__ == '__main__':

    args = docopt(__doc__)
    states = "Level1-1  Level2-1  Level3-1\
              Level4-1  Level5-1  Level6-1\
              Level7-1  Level8-1".split()
    t = 0
    epsilon = float(args['--epsilon'])
    strategy = None
    init_iterations = 20
    status = ""

    try:
        while True:
            emulator_state = states[
                np.random.choice(np.array(range(8)), 1, p=softmax(np.array([7, 6, 5, 4, 3, 2, 1, 0]) + 3))[0]
            ]
            if t:
                env.load_state(emulator_state)
            else:
                env = retro.make("SuperMarioBros-nes", emulator_state, scenario='scenario')
                show_all_actions_meaning(env)
                action_meaning = [str(env.get_action_meaning(action)) for action in meaningful_actions]
            next_state = env.reset()
            input_shape = pre_process(next_state).shape
            t = 0
            total_reward = 0
            if not strategy:
                strategy = DQLStrategy(env, number_of_actions=len(meaningful_actions), input_shape=input_shape)
                custom_epsilon_greedy(strategy, 0, pre_process(next_state))
            else:
                strategy.environment = env
            while True:
                state = env.get_screen()
                state = pre_process(state)

                action = custom_epsilon_greedy(strategy, epsilon, state)

                next_state, reward, done, info = env.step(meaningful_actions[action])
                next_state = pre_process(next_state)
                strategy.update(state, action, liveness(reward, meaningful_actions[action]), next_state, done)
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
                total_reward += liveness(reward, meaningful_actions[action])
                if done:
                    if not init_iterations:
                        epsilon *= 0.994  # decay epsilon at each episode
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
