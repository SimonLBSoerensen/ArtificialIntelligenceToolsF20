from ludpyhelper.RL.QLearning.qtable import QTable
import numpy as np
import bottleneck as bn
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

move_treshold = 200
epochs = 20_000

epsilon = 1.0
epsilon_decay_start = 1
epsilon_decay_end = epochs//7
epsilon_decay = epsilon / (epsilon_decay_end - epsilon_decay_start)


agent = QTable(env.action_space.n, epsilon=epsilon, epsilon_update=[epsilon_decay_start, epsilon_decay_end, epsilon_decay])


def make_state(observation):
    mins = np.array([-2.4, -10, -12, -10])
    maxs = np.array([2.4, 10, 12, 10])
    states_steps = np.array([40]*len(mins))

    norm_states = (np.array(observation) - mins) / (maxs - mins)
    norm_states = np.clip(norm_states, 0.0, 1.0)
    discrete_states = np.round(states_steps * norm_states).astype(int)
    return tuple(discrete_states)

reward_hist = []
epsilon_hist = []
for epoch in tqdm(range(epochs)):
    observation = env.reset()
    state = make_state(observation)
    done = False
    moves_count = 0
    epoch_reward = 0

    while not done:
        #env.render()

        action = agent.act(state)

        (observation, reward, done, info) = env.step(action) # take a random action
        new_state = make_state(observation)
        if done:
            reward = -1.0
        epoch_reward += reward
        agent.update(state, action, new_state, reward, done)

        state = new_state
        
        if move_treshold < moves_count:
            break
        moves_count+=1

    reward_hist.append(epoch_reward)
    epsilon_hist.append(agent.epsilon)
    pass
env.close()

plt.figure()
plt.title("Mean")
plt.plot(bn.move_mean(reward_hist, window=100, min_count=1))
plt.show()

plt.figure()
plt.title("Min")
plt.plot(bn.move_min(reward_hist, window=100, min_count=1))
plt.show()

plt.figure()
plt.title("Max")
plt.plot(bn.move_max(reward_hist, window=100, min_count=1))
plt.show()

plt.figure()
plt.title("Epsilon")
plt.plot(epsilon_hist)
plt.show()

pass

