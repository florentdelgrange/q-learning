import numpy as np

import retro
import random

from q_learning_strategies import RandomStrategy, DQL


def custom_epsilon_greedy(env, strategy, epsilon, state):
    p = random.random()
    if p < epsilon:
        p = random.random()
        if p < 0.2:
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        else:
            return env.action_space.sample()
    else:
        return strategy.play(state)


if __name__ == '__main__':
    states = "Bridges1          Forest1           VanillaDome5\
        Bridges2          Forest2           YoshiIsland1\
        ChocolateIsland1  Forest3           YoshiIsland2\
        ChocolateIsland2  Forest4           YoshiIsland3\
        ChocolateIsland3  Forest5           YoshiIsland4\
        DonutPlains1      Start             \
        DonutPlains2      VanillaDome1      \
        DonutPlains3      VanillaDome2      \
        DonutPlains4      VanillaDome3      \
        DonutPlains5      VanillaDome4".split()
    t = 0
    epsilon = 0.9
    strategy = None
    try:
        while True:
            state = states[random.randint(0, len(states) - 1)]
            if t:
                env.load_state(state)
            else:
                env = retro.make("SuperMarioWorld-Snes", state, scenario='scenario2')
            next_state = env.reset()
            input_shape = next_state.shape
            t = 0
            total_reward = 0
            if not strategy:
                # strategy = RandomStrategy(env)
                strategy = DQL(env)
            else:
                strategy.environment = env
            while True:
                state = env.get_screen()
                # action = strategy.play(state)  # actions are one hot encoded
                action = custom_epsilon_greedy(env, strategy, epsilon, state)
                next_state, reward, done, info = env.step(action)
                if done:
                    reward = -500.
                strategy.update(state, action, reward, next_state, done)
                t += 1
                if t % 10 == 0:
                    infostr = ''
                    if t % 100 == 0 and info:
                        infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                        print(('t=%i' % t) + infostr)
                    env.render()
                total_reward += reward
                if reward > 0:
                    print('t=%i got reward: %g, total reward: %g' % (t, reward, total_reward))
                if reward < 0:
                    print('t=%i got penalty: %g, total reward: %g' % (t, -reward, total_reward))
                if done:
                    epsilon *= 0.98  # decay epsilon at each episode
                    env.render()
                    try:
                        print("done! time=%i, reward=%d" % (t, total_reward))
                        input("press enter to continue")
                        print()
                    except EOFError:
                        exit(0)
                    break

    except KeyboardInterrupt:
        exit(0)
