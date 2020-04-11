from ludpyhelper.RL.QLearning.qtable import QTable, NQTable
from ludpyhelper.RL.helpers.reward_tracker import RTrack
import numpy as np
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')

load_old_table = True
epochs = 20_000
render_every = epochs//100

epsilon = 1.0
epsilon_decay_start = 1
epsilon_decay_end = (epochs // 4) * 3
epsilon_decay = epsilon / (epsilon_decay_end - epsilon_decay_start)

agent = NQTable(action_space=env.action_space.n, n_q_tabels=2, initial_memory_size=100, max_memory_size=4_000,
                n_old=75, k=20, epsilon=epsilon, epsilon_update=[epsilon_decay_start, epsilon_decay_end, epsilon_decay],
                learning_rate=0.1, discount_factor=0.95, q_init=0,
                shrinking_threshold=None, adaptively=True)
if load_old_table:
    agent.load_table("qTableLunarLander.pickel")


def make_state(observation):
    mins = np.full(env.observation_space.shape[0], -3)
    maxs = np.full(env.observation_space.shape[0], 3)
    states_steps = np.array([40] * len(mins))

    norm_states = (np.array(observation) - mins) / (maxs - mins)
    norm_states = np.clip(norm_states, 0.0, 1.0)
    discrete_states = np.round(states_steps * norm_states).astype(int)
    return tuple(discrete_states)


rt = RTrack()

there_has_been_a_winner = False
epsilon_hist = []
memory_size = []
for epoch in tqdm(range(epochs)):
    observation = env.reset()
    state = make_state(observation)
    done = False
    epoch_reward = 0

    while not done:
        if not epoch%render_every:
            env.render()

        action = agent.act(state)

        (observation, reward, done, info) = env.step(action)  # take a random action
        new_state = make_state(observation)

        epoch_reward += reward
        agent.update(state, action, new_state, reward, done)

        state = new_state

    agent.train_on_memory(100)

    if not there_has_been_a_winner and epoch_reward >= 200:
        print(f"Winner at: {epoch}")
        there_has_been_a_winner = True

    rt.add(epoch_reward)
    epsilon_hist.append(agent.epsilon)
    memory_size.append(len(agent.aer))
    pass

env.close()

rt.plot()

plt.figure()
plt.title("epsilon_hist")
plt.plot(epsilon_hist)
plt.show()

plt.figure()
plt.title("memory_size")
plt.plot(memory_size)
plt.show()

agent.save_table("qTableLunarLander.pickel")

pass
