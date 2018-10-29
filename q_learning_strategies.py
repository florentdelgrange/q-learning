from keras import Sequential, Input, Model
from keras.losses import mse

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout, Activation, concatenate, \
    AveragePooling2D
import numpy as np

LOGS = True


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
    x = Dense(1, name="q-values")(x)
    model = Model(inputs=[state_input, action_input], outputs=[x], name=name)

    model.compile(optimizer='nadam', loss=mse)

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
                self.buf = np.concatenate((buf[batch_size:],
                                           np.empty(shape=batch_size, dtype=np.object)), axis=0)
                self.length = max(self.length - batch_size, 0)
                self.index = (self.index - batch_size) % self.maxlen
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

    def __init__(self, environment, gamma=0.9, history_size=64, replay_memory_size=12000, switch_network_episode=5):
        super().__init__(environment)
        self.input_shape = environment.observation_space.shape
        self.main_dqn = dqn_init(self.input_shape, environment.action_space.shape, name="Main DQN")
        self.exploration_dqn = None
        self.__iteration = 0
        self.__episode = 0
        self.switch_network_episode = switch_network_episode
        self.history_size = history_size
        self.replay_memory = ReplayMemory(replay_memory_size)
        self.__random_exploration_phase = True
        self.gamma = gamma  # discount factor

        n = self.environment.action_space.shape[0]
        self.action_space = get_all_actions(n, environment)
        #    self.action_space = np.empty((n + 1, n), dtype='?')  # generate all possible actions
        #    for i in range(n + 1):
        #        for j in range(n):
        #            self.action_space[i][j] = i > j

    def play(self, state):
        state = pre_process_input_state(state)

        if self.exploration_dqn:
            return self.action_space[
                np.argmax(
                    [self.exploration_dqn.predict([state, pre_process_input_action(action)])
                     for action in self.action_space]
                )
            ]
        else:
            return self.environment.action_space.sample()

    def update(self, state, action, reward, next_state, done=False):
        self.replay_memory.append((state, action, reward, next_state))

        self.__iteration += 1
        if self.__random_exploration_phase:
            self.__iteration %= self.replay_memory.maxlen
        else:
            self.__iteration %= self.history_size

        if done:
            self.__episode += 1
            if LOGS:
                print("Episode {}, replay memory size: ".format(self.__episode, self.replay_memory.length))
            if not (self.__episode % self.switch_network_episode) and not self.__random_exploration_phase:
                if LOGS:
                    print("Episode {}: switching deep q-network...".format(self.__episode))
                self.exploration_dqn = \
                    dqn_init(self.input_shape, self.environment.action_space.shape[0],
                             name="Exploration DQN (episode {})".format(self.__episode))
                self.exploration_dqn.set_weights(self.main_dqn.get_weights())

        if not self.__iteration:
            self.__random_exploration_phase = False

            observations = [[], [], [], []]  # state, action, reward, next_state
            for memory in self.replay_memory.sample(self.history_size):
                for i, value in enumerate(memory):
                    observations[i].append(value)
            observations = [np.array(col) for col in observations]
            states, actions, rewards, next_states = \
                pre_process_input_state(observations[0]), pre_process_input_action(observations[1]), \
                observations[2], pre_process_input_state(observations[3])
            q_values = np.empty(shape=(len(states),), dtype='float64')
            for i in range(q_values.size):
                if self.exploration_dqn:
                    q_values[i] = rewards[i] + self.gamma * \
                                  np.max(self.exploration_dqn.predict(
                                      [next_states[i], action] for action in self.action_space))
                else:
                    q_values[i] = rewards[i]
            if LOGS:
                print("Fit critical deep q-network...")
            self.main_dqn.fit({'state': states, 'action': actions},
                              {'q-values': q_values},
                              epochs=1, batch_size=self.history_size)
