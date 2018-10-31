import random
import os

from keras import Sequential, Input, Model
from keras.callbacks import ModelCheckpoint
from keras.losses import mse
from keras.optimizers import SGD

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout, Activation, concatenate, \
    AveragePooling2D
import numpy as np

LOGS = True
CHECKPOINT_PATH = './models/model_checkpoint.hdf5'
CHECKPOINT = ModelCheckpoint(CHECKPOINT_PATH, verbose=1, save_best_only=False)
CALLBACKS = [CHECKPOINT]


def pre_process_input_state(s):
    s = np.array(s, dtype='float')
    s /= 255.
    s -= 0.5
    s *= 2.
    return s


def pre_process_input_action(a):
    a = np.array(a, dtype='float')
    a -= 0.5
    a *= 2.
    return a


def dqn_init(state_input_shape, action_input_shape, name="Deep-Q-Network"):
    state_input = Input(shape=state_input_shape, name="state")
    action_input = Input(shape=action_input_shape, name="action")
    x = Conv2D(16, kernel_size=(5, 5), strides=(1, 1),
               activation='relu',
               input_shape=state_input_shape)(state_input)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = concatenate([x, action_input])
    x = Dense(128, activation='relu')(x)
    x = Dense(1, kernel_initializer='zeros', name="q-values")(x)
    model = Model(inputs=[state_input, action_input], outputs=[x], name=name)

    model.compile(optimizer=SGD(lr=0.005, momentum=0.95, decay=0., nesterov=True), loss=mse)

    model.summary()
    return model


class Strategy:

    def __init__(self, environment):
        self.environment = environment

    def update(self, state, action, reward, next_state, done=False):
        pass

    def play(self, state):
        pass


class RandomStrategy(Strategy):

    def __init__(self, environment):
        super().__init__(environment)

    def play(self, state):
        return self.environment.action_space.sample()


class ReplayMemory:
    """
    Ring buffer used to remember observations during Q-learning
    """

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=False):
        """
        Gather n random samples from the ring buffer.
        :param batch_size: number of samples to randomly gather from the ReplayMemory
        :param with_replacement: (optional); if this value is set to True, the samples gathered are removed from the
                                 buffer.
        :return: random samples from the ReplayMemory
        """
        if with_replacement:
            indices = np.random.permutation(self.length)
            buf = self.buf[indices]
            if batch_size < self.length:
                rest = buf[batch_size:]
                self.buf = np.concatenate(
                    (rest, np.empty(shape=(batch_size + self.maxlen - self.length), dtype=np.object)),
                    axis=0)
                self.length = max(self.length - batch_size, 0)
                self.index = len(rest)
                return buf[:batch_size]
            else:
                self.buf = np.empty(shape=self.maxlen, dtype=np.object)
                self.length = 0
                self.index = 0
                return buf
        else:
            indices = np.random.randint(self.length, size=batch_size)  # faster
            return self.buf[indices]


def get_all_actions(n, env):
    """
    Generate all pertinent combinations (wrt. the environment) of actions for a given size of vector given
    Example :
        - 12 actions
        - Vectors are of the form [1, 0, 0, ..., 0], [0, 1, 0, ..., 0], etc. in one hot encoding
        - get_all_actions(12, env) will form all combinations of actions in 2**12 (e.g., [1, 0, 1, ..., 1]
        - get_all_actions(12, env) will ignore combinations of actions that have the same meaning

    :param n: number of actions
    :param env: environment
    :return: all pertinents combinations of actions
    """

    def generate_power_set(n):
        if n == 1:
            return [[1], [0]]
        else:
            action_space = []
            for suffix in generate_power_set(n - 1):
                action_space.append([0] + suffix)
                action_space.append([1] + suffix)
            return action_space

    power_set = np.array(generate_power_set(n), dtype='?')
    actions = []
    actions_meaning = set()

    def get_action_from_list(action_combination):
        action_str = ""
        for action in action_combination:
            action_str += action + " + "
        return action_str[:-3]

    for action in power_set:
        action_meaning = env.get_action_meaning(action)
        if get_action_from_list(action_meaning) not in actions_meaning:
            actions.append(action)
            actions_meaning.add(get_action_from_list(action_meaning))
    return np.array(actions, dtype='?')


class DQL(Strategy):

    def __init__(self, environment, gamma=0.99, batch_size=64, replay_memory_size=1200, switch_network_episode=16,
                 input_shape=None, action_space=[]):
        super().__init__(environment)
        if input_shape:
            self.input_shape = input_shape

        self.main_dqn = dqn_init(self.input_shape, environment.action_space.shape, name="Main DQN")
        self.exploration_dqn = dqn_init(self.input_shape, self.environment.action_space.shape,
                                        name="Exploration DQN (episode 0)")
        self.__random_exploration_phase = True

        if os.path.isfile(CHECKPOINT_PATH):
            self.__random_exploration_phase = False
            self.main_dqn.load_weights(CHECKPOINT_PATH)
            self.exploration_dqn.load_weights(CHECKPOINT_PATH)
            if LOGS:
                print("{}: weights loaded".format(CHECKPOINT_PATH))

        self.__iteration = 0
        self.__episode = 1
        self.switch_network_episode = switch_network_episode
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(replay_memory_size)
        self.gamma = gamma  # discount factor

        n = self.environment.action_space.shape[0]
        if len(action_space):
            self.action_space = action_space
        else:
            self.action_space = get_all_actions(n, environment)

    def play(self, state, exploration=True):

        if not exploration:
            return self.action_space[
                np.argmax(
                    self.main_dqn.predict(
                        {'state': pre_process_input_state([state] * len(self.action_space)),
                         'action': pre_process_input_action(self.action_space)}
                    )
                )
            ]
        elif self.__random_exploration_phase:
            return self.action_space[random.randint(0, self.action_space.shape[0] - 1)]

        else:
            return self.action_space[
                np.argmax(
                    self.exploration_dqn.predict(
                        {'state': pre_process_input_state([state] * len(self.action_space)),
                         'action': pre_process_input_action(self.action_space)}
                    )
                )
            ]

    def update(self, state, action, reward, next_state, done=False):
        self.replay_memory.append((state, action, reward, next_state))

        self.__iteration += 1
        self.__iteration %= self.replay_memory.maxlen

        if not self.__iteration:
            self.__random_exploration_phase = False

        if done and not self.__random_exploration_phase:
            self.__episode += 1
            if LOGS:
                print("Episode {}, replay memory size: {}".format(self.__episode, self.replay_memory.length))
            if not self.__episode % self.switch_network_episode:
                if LOGS:
                    print("Episode {}: exploration DQN <- copy of main DQN...".format(self.__episode))
                self.exploration_dqn.set_weights(self.main_dqn.get_weights())

        if not self.__iteration:
            self.fit_main_dqn()

    def fit_main_dqn(self):

        if LOGS:
            print("Fit critical deep q-network...")

        self.main_dqn.fit_generator(self.q_generator(),
                                    steps_per_epoch=self.replay_memory.length // self.batch_size,
                                    shuffle=False,
                                    callbacks=CALLBACKS)

    def q_generator(self):
        while True:
            n = self.replay_memory.length
            states_shape = [n] + list(self.input_shape)
            actions_shape = [n] + list(self.action_space.shape)
            observations = [np.empty(shape=states_shape, dtype='uint8'),  # state
                            np.empty(shape=actions_shape, dtype='?'),     # action
                            np.empty(shape=n, dtype='float'),             # reward
                            np.empty(shape=states_shape, dtype='uint8')   # next_state
                            ]
            for j, memory in enumerate(self.replay_memory.sample(n)):
                for i, value in enumerate(memory):
                    observations[i][j] = value
            observations = [np.array(col) for col in observations]
            states = observations[0]
            actions = observations[1]
            rewards = observations[2]
            next_states = observations[3]
            epochs = n // self.batch_size
            index = lambda x, y: x * self.batch_size + y
            for i in range(epochs):
                q_values = np.empty(shape=self.batch_size, dtype='float')
                for j in range(self.batch_size):
                    successor = pre_process_input_state(
                        [next_states[index(i, j)] * len(self.action_space)]
                    )
                    values_next_state = np.max(
                        self.exploration_dqn.predict(
                            {'state': successor,
                             'action': self.action_space}
                        )
                    )
                    #  q(states[i], actions[i]) of the current observation i
                    q_values[j] = rewards[index(i, j)] + self.gamma * values_next_state
                yield {'state': pre_process_input_state(states[index(i, 0): index(i + 1, 0)]),
                       'action': pre_process_input_action(actions[index(i, 0): index(i + 1, 0)])
                       }, {'q-values': q_values}
