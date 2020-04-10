from ludpyhelper.RL.QLearning.qtable import QTable
from ludpyhelper.RL.helpers.reward_tracker import RTrack
import numpy as np
import bottleneck as bn
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

load_old_table = False
epochs = 25_000

epsilon = 1.0
epsilon_decay_start = 1
epsilon_decay_end = epochs // 4 * 3
epsilon_decay = epsilon / (epsilon_decay_end - epsilon_decay_start)

agent = QTable(env.action_space.n, epsilon=epsilon,
               epsilon_update=[epsilon_decay_start, epsilon_decay_end, epsilon_decay])
if load_old_table:
    agent.load_table("qTable_MountainCar.pickel")


def make_state(observation):
    mins = np.array([-1.2, -0.07])
    maxs = np.array([0.6, 0.07])
    states_steps = np.array([30] * len(mins))

    norm_states = (np.array(observation) - mins) / (maxs - mins)
    norm_states = np.clip(norm_states, 0.0, 1.0)
    discrete_states = np.round(states_steps * norm_states).astype(int)
    return tuple(discrete_states)


def goal_check(observation):
    position, velocity = observation
    goal = position >= env.env.goal_position and velocity >= env.env.goal_velocity
    return goal


rt = RTrack()

epsilon_hist = []
for epoch in tqdm(range(epochs)):
    observation = env.reset()
    state = make_state(observation)
    done = False
    epoch_reward = 0

    while not done:
        # env.render()

        action = agent.act(state)

        (observation, reward, done, info) = env.step(action)  # take a random action
        new_state = make_state(observation)
        if goal_check(observation):
            reward = 0
        epoch_reward += reward
        agent.update(state, action, new_state, reward, done)

        state = new_state

    rt.add(epoch_reward)
    epsilon_hist.append(agent.epsilon)
    pass
env.close()

rt.plot()

plt.figure()
plt.plot(epsilon_hist)
plt.show()

agent.save_table("qTable_MountainCar.pickel")

pass

