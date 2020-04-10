from ludpyhelper.RL.QLearning.qtable import QTable, NQTable
from ludpyhelper.RL.helpers.reward_tracker import RTrack
import numpy as np
import bottleneck as bn
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

load_old_table = False
move_treshold = 200
epochs = 100_000

epsilon = 1.0
epsilon_decay_start = 1
epsilon_decay_end = epochs//4 * 3
epsilon_decay = epsilon*0.8 / (epsilon_decay_end - epsilon_decay_start)


agent = NQTable(env.action_space.n, 2, epsilon=epsilon, learning_rate=0.2, epsilon_update=[epsilon_decay_start, epsilon_decay_end, epsilon_decay])
if load_old_table:
    agent.load_table("qTable.pickel")


def make_state(observation):
    mins = np.array([-2.4, -10, -12, -10])
    maxs = np.array([2.4, 10, 12, 10])
    states_steps = np.array([40]*len(mins))

    norm_states = (np.array(observation) - mins) / (maxs - mins)
    norm_states = np.clip(norm_states, 0.0, 1.0)
    discrete_states = np.round(states_steps * norm_states).astype(int)
    return tuple(discrete_states)


rt = RTrack()

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

    rt.add(epoch_reward)
    epsilon_hist.append(agent.epsilon)
    pass
env.close()

rt.plot()

plt.figure()
plt.plot(epsilon_hist)
plt.show()

agent.save_table("qTable.pickel")

pass

