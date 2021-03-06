import random
import os

import tensorflow as tf
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.losses import mse
from keras.optimizers import SGD, RMSprop, Nadam, Adam

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout, Activation, concatenate, \
    AveragePooling2D, Lambda, Multiply
from keras import backend as K
import numpy as np
from keras.layers.normalization import BatchNormalization

from SarstReplayMemory import SarstReplayMemory

global graph
graph = tf.get_default_graph()

CHECKPOINT_PATH = './models/model_checkpoint.hdf5'


def pre_process_input_state(s):
    s /= 255.
    s -= 0.5
    s *= 2.
    return s


def action_one_hot_encoder(number_of_actions, action):
    one_hot_vector = np.zeros(shape=number_of_actions, dtype='?')
    one_hot_vector[action] = 1
    return one_hot_vector


def huber_loss(y_true, y_pred, clip_value=1.0):
    assert clip_value > 0.
    x = y_true - y_pred
    if np.isinf(clip_value):
        return .5 * K.square(x)
    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if hasattr(tf, 'select'):
        return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
    else:
        return tf.where(condition, squared_loss, linear_loss)  # condition, true, false


def dqn_init(state_input_shape, number_of_actions, name="Deep-Q-Network"):
    state_input = Input(shape=state_input_shape, name="state")
    x = Lambda(pre_process_input_state)(state_input)
    x = Conv2D(8, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (5, 5), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(number_of_actions, name="q-values", kernel_initializer='zeros', activation='linear')(x)

    action_input = Input(shape=(number_of_actions,), name="action_mask")
    output = Multiply(name="q-values-masked")([x, action_input])

    model = Model(inputs=[state_input, action_input], outputs=output, name=name)

    #   model.compile(optimizer=SGD(lr=0.002, momentum=0.95, decay=0., nesterov=True), loss=mse)
    model.compile(optimizer=Adam(lr=0.0003), loss=mse)
    #   learning_rate_decay = 0.999999
    #   model.compile(optimizer=RMSprop(lr=0.0003, rho=0.95, epsilon=0.01, decay=learning_rate_decay), loss=huber_loss)

    model.summary()
    return model


class Strategy:

    def __init__(self, environment):
        self.environment = environment

    def update(self, state, action, reward, next_state, done=False):
        pass

    def play(self, state):
        pass

    @property
    def logs(self):
        return ""


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

    def sample(self, batch_size, with_replacement=True):
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
    Generate all pertinent combinations (w.r.t. the environment) of actions for a given size of vector given
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


class DQLStrategy(Strategy):

    def __init__(self, environment, gamma=0.92,
                 batch_size=32, replay_memory_size=51200, history_size=12800,
                 switch_network_episode=8, input_shape=None, number_of_actions=0,
                 checkpoint_path="", current_episode=1):
        super().__init__(environment)

        self.__logs = ''  # gather logs during each iteration

        if input_shape:
            self.input_shape = input_shape
        else:
            self.input_shape = environment.observation_space.shape

        if number_of_actions:
            self.number_of_actions = number_of_actions
        else:
            n = self.environment.action_space.shape[0]
            self.number_of_actions = len(get_all_actions(n, environment))

        self.main_dqn = dqn_init(self.input_shape, self.number_of_actions, name="Main DQN")
        self.exploration_dqn = dqn_init(self.input_shape, self.number_of_actions, name="Exploration DQN (episode 0)")
        self.__random_exploration_phase = True
        self.__weights_loaded = False

        if checkpoint_path:
            self.main_dqn.load_weights(checkpoint_path)
            self.exploration_dqn.load_weights(checkpoint_path)
            self.__weights_loaded = True
            self.__logs += "{}: weights loaded ".format(checkpoint_path)

        elif os.path.isfile(CHECKPOINT_PATH):
            self.main_dqn.load_weights(CHECKPOINT_PATH)
            self.exploration_dqn.load_weights(CHECKPOINT_PATH)
            self.__weights_loaded = True
            self.__logs += "{}: weights loaded ".format(CHECKPOINT_PATH)

        self.__iteration = 0
        self.__episode = current_episode
        self.switch_network_episode = switch_network_episode
        self.batch_size = batch_size
        self.history_size = history_size
        self.replay_memory = SarstReplayMemory(replay_memory_size, input_shape)
        self.gamma = gamma  # discount factor
        self.exploration = True

    def play(self, state):
        state = np.array([state])
        ones = np.array([np.ones(shape=self.number_of_actions)])

        if not self.exploration:
            return np.argmax(
                self.main_dqn.predict([state, ones])[0]
            )
        elif self.__random_exploration_phase and not self.__weights_loaded:
            return random.randint(0, self.number_of_actions - 1)

        else:
            return np.argmax(
                self.exploration_dqn.predict([state, ones])[0]
            )

    def get_q_values(self, state):
        state = np.array([state])
        ones = np.array([np.ones(shape=self.number_of_actions)])

        if self.exploration:
            return self.exploration_dqn.predict([state, ones])[0]
        else:
            return self.main_dqn.predict([state, ones])[0]

    def q_value(self, state, action):
        state = np.array([state])
        action = np.array([action_one_hot_encoder(self.number_of_actions, action)])

        if self.exploration:
            return self.exploration_dqn.predict([state, action])[0]
        else:
            return self.main_dqn.predict([state, action])[0]

    def update(self, state, action, reward, next_state, done=False):
        self.replay_memory.append(state, action, reward, next_state, done)

        self.__iteration += 1

        if self.__random_exploration_phase:
            self.__iteration %= self.replay_memory.memory_capacity
        else:
            self.__iteration %= self.history_size

        if not self.__iteration:
            self.__episode += 1
            self.fit_main_dqn()
            self.__logs += "Episode {} ".format(self.__episode)
            if self.__random_exploration_phase:
                self.__random_exploration_phase = False
                self.__logs += ": exploration DQN <- copy of main DQN... ".format(self.__episode)
                self.exploration_dqn.set_weights(self.main_dqn.get_weights())
            elif not self.__episode % self.switch_network_episode:
                self.__logs += ": exploration DQN <- copy of main DQN... ".format(self.__episode)
                self.exploration_dqn.set_weights(self.main_dqn.get_weights())

    def fit_main_dqn(self):
        print("Fit critical deep q-network...")
        if not self.__episode % self.switch_network_episode:
            CHECKPOINT = ModelCheckpoint("{}_episode{}.hdf5".format(CHECKPOINT_PATH[:-5], self.__episode), verbose=1,
                                         save_best_only=False)
            CALLBACKS = [CHECKPOINT]
            self.__logs += "weights saved; "
        else:
            CALLBACKS = []
        self.main_dqn.fit_generator(self.q_generator(),
                                    steps_per_epoch=(self.history_size // self.batch_size),
                                    callbacks=CALLBACKS)

    def q_generator(self):
        while True:
            n = self.history_size
            states, actions, rewards, next_states, done = self.replay_memory.sample(n)
            steps = n // self.batch_size
            index = lambda x, y: x * self.batch_size + y
            for i in range(steps):
                q_values = np.empty(shape=(self.batch_size, self.number_of_actions), dtype='float')
                actions_one_hot = np.empty(shape=(self.batch_size, self.number_of_actions), dtype='?')
                for j in range(self.batch_size):
                    action = actions[index(i, j)]
                    actions_one_hot[j] = action_one_hot_encoder(self.number_of_actions, action)
                    successor = np.array([next_states[index(i, j)]])
                    with graph.as_default():  # correct some synchronisation issues
                        q_values[j] = np.zeros(shape=self.number_of_actions)
                        if not done[index(i, j)]:
                            successor_q_values = self.exploration_dqn.predict(
                                [successor, np.array([np.ones(shape=self.number_of_actions)])]
                            )[0]
                            q_values[j][action] = rewards[index(i, j)] + self.gamma * np.amax(successor_q_values)
                            # with open("logs_training.log", "a+") as f:
                            #     f.write("current action={}, (one hot)={}, q_values(s, •)={}; r(s, {}, s')={}; max Q(s, a)={}; new q_values={}\n".format(action, str(actions_one_hot[j]), str(successor_q_values), action, rewards[index(i, j)], np.amax(successor_q_values), q_values[j]))
                        else:
                            q_values[j][action] = rewards[index(i, j)]

                yield [states[i * self.batch_size: (i + 1) * self.batch_size], actions_one_hot], q_values

    @property
    def logs(self):
        logs = self.__logs
        self.__logs = ""
        return logs
