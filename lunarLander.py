from ludpyhelper.RL.Deep import DQ
from ludpyhelper.RL.helpers.reward_tracker import RTrack
from ludpyhelper.mics.functions import ramp
import numpy as np
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow import keras

env = gym.make('LunarLander-v2')

load_old_table = True
epochs = 5_000
render_every = epochs // 100

epsilon = 1.0
epsilon_decay_start = 1
epsilon_decay_end = (epochs // 4) * 3

def make_state(observation):
    mins = np.full(env.observation_space.shape[0], -3)
    maxs = np.full(env.observation_space.shape[0], 3)

    norm_states = (np.array(observation) - mins) / (maxs - mins)
    norm_states = np.clip(norm_states, 0.0, 1.0)
    return norm_states

def creat_model_func():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(16, activation="relu", input_shape=env.observation_space.shape))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(env.action_space.n))

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    return model


def epsilon_update(ep):
    return ramp(ep, 1.0, epsilon_decay_end, epsilon_decay_start)


agent = DQ(creat_model_func=creat_model_func, log_dir="Logs", update_target_every=5, initial_memory_size=100,
           max_memory_size=4_000,
           n_old=75, k=20, epsilon_update=epsilon_update,
           discount_factor=0.95,
           shrinking_threshold=None, adaptively=True)

rt = RTrack()

there_has_been_a_winner = False
epsilon_hist = []
memory_size = []
for epoch in tqdm(range(epochs)):
    state = make_state(env.reset())
    done = False
    epoch_reward = 0

    while not done:
        if not epoch % render_every:
            env.render()

        action = agent.act(state)

        (new_state, reward, done, info) = env.step(action)
        new_state = make_state(new_state)

        epoch_reward += reward
        agent.add_to_pre_memory(state, action, new_state, reward, done)

        state = new_state

    agent.train_on_memory(64, 64, True, flat_sample=False, update_memory=True)

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

pass
