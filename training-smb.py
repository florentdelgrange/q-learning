"""Deep q-learning training for Super Mario Bros (NES)

Usage:
    training.py [--epsilon <epsilon>] [--episode <episode>] [--model_path <path>]
    training.py (-h | --help)

Options:
-h --help                           Display help.
-e --epsilon <epsilon>              Actions are selected randomly with a probability epsilon (epsilon-greedy algorithm). [default: 1.]
-p --model_path <path>              Path of weights to load.
--episode <episode>                 Episode. [default: 1]
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
    # [1., 0., 0., 0., 0., 0., 0., 0., 0.],  # B button
    # [0., 0., 0., 0., 1., 0., 0., 0., 0.],  # UP button
    [0., 0., 0., 0., 0., 1., 0., 0., 0.],  # DOWN button
    # [0., 0., 0., 0., 0., 0., 1., 0., 0.],  # LEFT button
    # [0., 0., 0., 0., 0., 0., 0., 1., 0.],  # RIGHT button
    # [0., 0., 0., 0., 0., 0., 0., 0., 1.],  # A button
    [1., 0., 0., 0., 0., 0., 0., 1., 0.],  # B + RIGHT
    # [1., 0., 0., 0., 0., 0., 0., 1., 1.],  # A + B + RIGHT
    # [0., 0., 0., 0., 1., 0., 0., 0., 1.],  # A + UP
    # [0., 0., 0., 0., 1., 0., 0., 1., 1.],  # A + UP  + RIGHT
    [1., 0., 0., 0., 1., 0., 0., 1., 1.],  # A + B + UP + RIGHT
    [1., 0., 0., 0., 0., 0., 1., 0., 0.],  # B + LEFT
    # [1., 0., 0., 0., 0., 0., 1., 0., 1.],  # A + B + LEFT
    # [0., 0., 0., 0., 1., 0., 1., 0., 1.],  # A + UP  + LEFT
    [1., 0., 0., 0., 1., 0., 1., 0., 1.],  # A + B + UP + LEFT
    # [0., 0., 0., 0., 0., 1., 0., 1., 0.],  # DOWN + RIGHT
    # [0., 0., 0., 0., 0., 1., 1., 0., 0.],  # DOWN + LEFT
    # [0., 0., 0., 0., 0., 0., 0., 0., 0.],  # do nothing
], dtype='?')  # meaningful actions for SuperMarioWorld

action_meaning = None

align = lambda x: x if len(x) == 2 else x + " "
space = lambda x: " " * 8 if x >= 0 else " " * 7

current_LOGS = ""
ERASE_LINE = '\x1b[2K'
CURSOR_UP_ONE = '\x1b[1A'
TEMPORAL_MEMORY = 4


def softmax(x, temperature=1e-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

def custom_epsilon_greedy(strategy, epsilon, state, current_reward=0, max_ratio=1.):
    p = random.random()
    if p <= epsilon:
        return random.randint(0, meaningful_actions.shape[0] - 1)
    else:
        q_values = strategy.get_q_values(state)
        best_action = np.argmax(q_values)
        if not q_values.max():
            proba_on_q_values = np.ones(len(q_values)) / len(q_values)
        else:
            # choose best actions w.r.t. q-values
            if q_values.max() > 0:
                    best_actions = np.array(range(len(q_values)))[(q_values / q_values.max()) >= max_ratio]
                    # choose best q-values
                    best_q_values = q_values[(q_values / q_values.max()) >= max_ratio]
            else:
                    best_actions = np.array(range(len(q_values)))[(q_values.max() / q_values) >= max_ratio]
                    # choose best q-values
                    best_q_values = q_values[(q_values.max() / q_values) >= max_ratio]
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
            logs += "\nBest Action: {} | Action chosen: {}\n".format(best_action, action_meaning[choice])
            current_LOGS = logs
            return best_action


def pre_process(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (128, 128)), cv2.COLOR_RGB2GRAY)
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

def is_relevant_action(prev_state, next_state):
    for i in range(TEMPORAL_MEMORY):
        for j in range(TEMPORAL_MEMORY):
            if i != j and np.array_equal(prev_state[i][32:,], next_state[j][32:,]):
                return False
    return True

if __name__ == '__main__':

    args = docopt(__doc__)
    states = "Level1-1  Level2-1  Level3-1\
              Level4-1  Level5-1  Level6-1\
              Level7-1  Level8-1".split()
    t = 0
    epsilon = float(args['--epsilon'])
    initial_episode = int(args['--episode'])
    model_path = args['--model_path'] if args['--model_path'] else ""
    strategy = None
    init_iterations = 10
    status = ""
    episode = 0
    max_ratio = min([1, epsilon * 1.5])
    best_score = None
    prev_state = None

    try:
        while True:
            emulator_state = states[
                np.random.choice(np.array(range(8)), 1, p=softmax(np.array([7, 6, 5, 4, 3, 2, 1, 0]) + 3, temperature=1/(2*epsilon)))[0]
            ]
            if t:
                env.load_state(emulator_state)
            else:
                env = retro.make("SuperMarioBros-nes", emulator_state, scenario='scenario')
                action_meaning = [str(env.get_action_meaning(action)) for action in meaningful_actions]
            next_state = env.reset()
            next_state = pre_process(next_state)
            next_state = np.array([next_state] * TEMPORAL_MEMORY)
            next_state = np.stack(next_state, -1)
            input_shape = next_state.shape
            t = 0
            total_reward = 0
            if not strategy:
                strategy = DQLStrategy(env, number_of_actions=len(meaningful_actions), input_shape=input_shape, checkpoint_path=model_path, current_episode=initial_episode)
                custom_epsilon_greedy(strategy, 0, next_state, max_ratio=max_ratio)
            else:
                strategy.environment = env

            while True:
                state = next_state
                action = custom_epsilon_greedy(strategy, epsilon, state, current_reward=total_reward, max_ratio=max_ratio)

                next_state = np.empty(TEMPORAL_MEMORY, dtype='object')
                reward = 0
                done = False
                dead = False
                for i in range(TEMPORAL_MEMORY):
                    next_state_t, reward_t, done_t, info = env.step(meaningful_actions[action])
                    # Dying
                    if info['PlayerState'] in [6, 11] and not info['time'] == 0:
                        dead = True
                        reward_t = -0.5
                    # Pipe
                    if info['PlayerState'] in [3, 2]:
                        reward_t += 0.05
                    # Climbing vine
                    elif info['PlayerState'] == 1:
                        reward_t += 1.25
                    # Blocked
                    elif info['PlayerState'] in [9, 12]:
                        reward_t = -0.05
                    reward += reward_t
                    next_state[i] = pre_process(next_state_t)
                    done = done_t or done
                # check if the action played is relevant
                # if prev_state != None and not is_relevant_action(prev_state, next_state):
                #     reward = -5
                    #for i, array in enumerate(list(prev_state) + list(next_state)):
                    #    plt.imsave("last_state{}.png".format(i), np.array(np.squeeze(array[32:,])))
                # prev_state = next_state

                next_state = np.stack(next_state, axis=-1)

                #  liveness
                liveness_reward = -4 / 375 if not reward and not dead else 0
                # right bonus
                if meaningful_actions[action][7] and not dead and reward > 0:
                    liveness_reward += 0.5
                # left and down malus
                if (meaningful_actions[action][6] or meaningful_actions[action][5]) and not dead and reward <= 0:
                    liveness_reward -= 0.5
                reward += liveness_reward

                strategy.update(state, action, reward, next_state, done)
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
                    sys.stdout.write("high score={} | current score={:9.3f} | current rewards={:9.3f}\n".format(best_score, info['score'], total_reward))
                    sys.stdout.write(current_LOGS)
                    sys.stdout.flush()
                total_reward += reward
                if done:
                    if best_score == None:
                        best_score = info['score']
                    else:
                        best_score = best_score if best_score >= info['score'] else info['score']
                    if not init_iterations:
                        episode += 1
                        epsilon = max(1e-3, epsilon * 0.995)  # decay epsilon at each episode
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
