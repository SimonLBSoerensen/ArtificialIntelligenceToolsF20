import numpy as np
from collections.abc import Iterable, Callable
from ludpyhelper.RL.helpers.replay_buffer import AER
from ludpyhelper.RL.Deep.tb import ModifiedTensorBoard
from tensorflow import keras
import os

class DQ:
    def __init__(self, creat_model_func, log_dir, update_target_every, initial_memory_size, max_memory_size,
                 n_old, k,  epsilon_update,
                 discount_factor=0.95,
                 shrinking_threshold=None, adaptively=True):
        self.discount_factor = discount_factor
        self.epsilon_update = epsilon_update
        self.aer = AER(initial_memory_size, max_memory_size, n_old, k, shrinking_threshold, adaptively)

        self.learning_model = creat_model_func()
        self.target_model = creat_model_func()
        self.target_model.set_weights(self.learning_model.get_weights())

        self.log_dir = log_dir
        self.update_target_every = update_target_every

        self.episode = 1

        self.target_update_counter = 0
        self.run_counter = 0

        self.pre_memory_buffer = []

    def _random_action(self):
        action = np.random.randint(0, self.learning_model.output.shape.as_list()[-1])
        return action

    def cal_q_values(self, state, model="learning"):
        state = np.array(state)
        rstate = state.reshape(-1, *state.shape)
        if model == "target":
            q_values = self.target_model.predict(rstate)
        else:
            q_values = self.learning_model.predict(rstate)
        return q_values

    def _epsilon_update(self):
        if self.epsilon_update is not None:
            self.epsilon = self.epsilon_update(self.episode)

    def act(self, state):
        self._epsilon_update()
        if np.random.random_sample() > self.epsilon:
            q_values = self.cal_q_values(state)
            action = np.argmax(q_values)
        else:
            action = self._random_action()
        return action

    def cal_target(self, future_qs, reward):
        return reward + self.discount_factor * np.max(future_qs)

    def add_to_pre_memory(self, current_state, action, new_state, reward, terminal_state):
        self.pre_memory_buffer.append([current_state, action, new_state, reward, terminal_state])

    def _handle_list_new_states(self, new_states):
        n = new_states.shape[0]
        list_l = new_states.shape[1]
        state_shape = new_states.shape[2:]
        new_states = new_states.reshape(-1, *state_shape)
        new_qss = self.target_model.predict(new_states)
        action_space = new_qss.shape[1]
        new_qss = new_qss.reshape(n, list_l * action_space)
        idxs = new_qss.argmax(axis=1) % list_l
        new_qss = np.array([p[i] for p, i in zip(new_qss, idxs)])
        return new_qss


    def update_memory(self, new_state_are_in_list=False):
        if len(self.pre_memory_buffer):
            current_states = np.array([el[0] for el in self.pre_memory_buffer])
            new_states = np.array([el[2] for el in self.pre_memory_buffer])

            current_qs = self.learning_model.predict(current_states)

            if new_state_are_in_list:
                new_qss = self._handle_list_new_states(new_states)
            else:
                new_qss = self.target_model.predict(new_states)

            for idx, (current_state, action, new_state, reward, terminal_state) in enumerate(self.pre_memory_buffer):
                current_q = current_qs[idx][action]
                new_qs = new_qss[idx]
                td = self.cal_target(new_qs, reward) - current_q
                self.aer.append([current_state, action, new_state, reward, terminal_state], td)

            self.pre_memory_buffer = []

    def update_target_model(self):
        self.target_model.set_weights(self.learning_model.get_weights())

    def train_on_memory(self, batch_size, fit_batch_size, terminal_state, flat_sample=False, update_memory=True, new_state_are_in_list=False):
        if update_memory:
            self.update_memory()
        batch = self.aer.sample(batch_size, flat_sample=flat_sample)

        current_states = np.array([el[0] for el in batch])
        new_states = np.array([el[2] for el in batch])

        current_states_qs = self.learning_model.predict(current_states)
        if new_state_are_in_list:
            new_states_qs = self._handle_list_new_states(new_states)
        else:
            new_states_qs = self.target_model.predict(new_states)

        batch_y = []

        for idx, (current_state, action, new_state, reward, terminal_state) in enumerate(batch):
            if not terminal_state:
                future_qs = new_states_qs[idx]
                target_q = self.cal_target(future_qs, reward)
            else:
                target_q = reward

            current_state_q_update = current_states_qs[idx]
            current_state_q_update[action] = target_q

            batch_y.append(current_state_q_update)

        batch_x = np.array(current_states)
        batch_y = np.array(batch_y)

        callbacks = [keras.callbacks.CSVLogger(os.path.join(self.log_dir, "run_log.csv"), append=True)]

        self.learning_model.fit(batch_x, batch_y, batch_size=fit_batch_size,
                                verbose=0, shuffle=False, callbacks=callbacks,
                                initial_epoch=self.run_counter, epochs=self.run_counter+1)
        self.run_counter += 1
        if terminal_state:
            self.target_update_counter += 1
            self.episode += 1

        if self.target_update_counter > self.update_target_every:
            self.update_target_model()
            self.target_update_counter = 0
