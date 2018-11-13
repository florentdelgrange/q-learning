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
    [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # UP + B button (leave water)
    [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],  # UP + A button (leave water)
    [1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # Y + B + RIGHT (accelerate RIGHT and jump)
    [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # Y + RIGHT (accelerate RIGHT)
    [0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],  # Y + RIGHT + DOWN (accelerate RIGHT and DOWN)
    [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],  # Y + A + RIGHT (accelerate RIGHT and spin jump)
    [1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # Y + B + LEFT (accelerate LEFT and jump)
    [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # Y + LEFT (accelerate LEFT)
    [0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],  # Y + LEFT (accelerate LEFT and DOWN)
    [0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],  # Y + A + LEFT (accelerate LEFT and spin jump)
    [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # B button + RIGHT
    [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # B button + LEFT
    [1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],  # UP + B button + RIGHT 
    [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],  # UP + B button + LEFT
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]   # Do nothing
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
    "UP + B button",
    "UP + A button",
    "Y + B + RIGHT",
    "Y + RIGHT",
    "Y + RIGHT + DOWN",
    "Y + A + RIGHT",
    "Y + B + LEFT",
    "Y + LEFT",
    "Y + LEFT + DOWN",
    "Y + A + LEFT",
    "B + RIGHT",
    "B + LEFT",
    "B + UP + RIGHT",
    "B + UP + LEFT",
    "Do nothing"
]

align = lambda x: x if len(x) == 2 else x + " "

current_LOGS = ""
ERASE_LINE = '\x1b[2K'
CURSOR_UP_ONE = '\x1b[1A'
TEMPORAL_MEMORY = 4


def softmax(x, temperature=0.1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)  # only difference
    #  return np.exp(x / temperature) / np.sum(np.exp(x / temperature))
            

LAST_ACTION = None
LAST_ACTION_COUNTER = 0


def custom_epsilon_greedy(strategy, epsilon, state, current_reward=0, max_ratio=1.):

    global LAST_ACTION
    global LAST_ACTION_COUNTER
    if LAST_ACTION:
        action = LAST_ACTION
        LAST_ACTION_COUNTER -= 1
        if not LAST_ACTION_COUNTER:
            LAST_ACTION = None
        return action

    p = random.random()
    if p < epsilon:
        if p >= epsilon / 8 or current_reward > -200 or epsilon >= 0.5:
            return random.randint(0, meaningful_actions.shape[0] - 1)
        else:
            #  play the same random action multiple times (four times in a row) with a probability epsilon / 8
            #  allows to unblock Mario in certain situations
            LAST_ACTION = random.randint(0, meaningful_actions.shape[0] - 1)
            LAST_ACTION_COUNTER = 4
            return LAST_ACTION
    else:
        q_values = strategy.get_q_values(state)
        best_action = np.argmax(q_values)
        if not q_values.max():
            proba_on_q_values = np.ones(len(q_values)) / len(q_values)
        else:
            # choose best actions w.r.t. q-values
            best_actions = np.array(range(len(q_values)))[(q_values / q_values.max()) >= max_ratio]
            # choose best q-values
            best_q_values = q_values[(q_values / q_values.max()) >= max_ratio]
            #  softmax on best q-values
            proba = softmax(best_q_values)
            proba_on_q_values = np.zeros(shape=len(q_values))
            for i, a in enumerate(best_actions):
                proba_on_q_values[a] = proba[i]

        #  logs
        logs = 7 * " " + "Q-values" + 8 * " " + "Softmax" + 7 * " " + "Action meaning\n"
        logs += "\n".join(
            [align(str(i)) + ": " + "{:11.6f}".format(x) + 8 * " " + "{:.4f}".format(proba_on_q_values[i]) + " " * 8 +
             action_meaning[i] for i, x in enumerate(q_values)]
        )
        #  softmax choice
        global current_LOGS
        if not q_values.max():
            logs += "\nBest Action: {}\n".format(" / ")
            current_LOGS = logs
            return random.randint(0, meaningful_actions.shape[0] - 1)
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
    # return np.reshape(observation, (84, 84, 1))
    return observation


def show_all_actions_meaning(env):
    unique_actions = list(range(env.action_space.n))
    for action in to_categorical(unique_actions):
        print(action)
        print(env.get_action_meaning(action))
        print(" ")


def liveness(reward, action):
    if action[7] and not action[5] and reward >= 0:
        return reward + 0.5
    elif action[7] and action[5] and reward >= 0:
        return reward
    elif not reward:
        return -0.5
    else:
        return reward


if __name__ == '__main__':

    args = docopt(__doc__)
    states_init = "             Forest1\
        Bridges2      \
        ChocolateIsland3  Forest5\
        DonutPlains1      Start             \
        DonutPlains2".split()
    states = "Bridges1          Forest2           YoshiIsland2\
        Bridges2          Forest3\
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
    init_iterations = 64
    status = ""
    episode = 0
    max_ratio = min([1, epsilon * 1.5])

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
            next_state = pre_process(next_state)
            next_state = np.array([next_state] * TEMPORAL_MEMORY)
            next_state = np.stack(next_state, -1)
            input_shape = next_state.shape
            t = 0
            total_reward = 0
            if not strategy:
                strategy = DQLStrategy(env, number_of_actions=len(meaningful_actions), input_shape=input_shape)
                custom_epsilon_greedy(strategy, 0, next_state, max_ratio=max_ratio)
            else:
                strategy.environment = env

            while True:
                state = next_state
                action = custom_epsilon_greedy(strategy, epsilon, state, current_reward=total_reward, max_ratio=max_ratio)

                next_state = np.empty(TEMPORAL_MEMORY, dtype='object')
                reward = 0
                done = False
                for i in range(TEMPORAL_MEMORY):
                    next_state_t, reward_t, done_t, _ = env.step(meaningful_actions[action])
                    reward += reward_t
                    next_state[i] = pre_process(next_state_t)
                    done = done_t or done
                next_state = np.stack(next_state, axis=-1)

                strategy.update(state, action, liveness(reward, meaningful_actions[action]), next_state, done)
                t += 1
                if not t % (10 // TEMPORAL_MEMORY):
                    env.render()
                    for _ in range(9 + len(meaningful_actions)):
                        sys.stdout.write(CURSOR_UP_ONE)
                        sys.stdout.write(ERASE_LINE)
                    strategy_logs = strategy.logs
                    status = strategy_logs if strategy_logs else status
                    sys.stdout.write("Emulator state: {}\n".format(emulator_state))
                    sys.stdout.write("Status: {}\n".format(status))
                    sys.stdout.write("ε={}\n".format(epsilon))
                    sys.stdout.write("best q-values ratio (for softmax): {}\n".format(max_ratio))
                    sys.stdout.write("current score={}\n".format(total_reward))
                    sys.stdout.write(current_LOGS)
                    sys.stdout.flush()
                total_reward += liveness(reward, meaningful_actions[action])
                if done:
                    if not init_iterations:
                        episode += 1
                        epsilon *= 0.99999  # decay epsilon at each episode
                        max_ratio = min([1, epsilon * 1.5])
                    else:
                        init_iterations -= 1
                    env.render()
                    try:
                        pass
                    except EOFError:
                        exit(0)
                    break

    except KeyboardInterrupt:
        exit(0)
