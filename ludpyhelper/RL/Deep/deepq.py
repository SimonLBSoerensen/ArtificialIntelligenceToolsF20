import numpy as np
from collections.abc import Iterable, Callable
from ludpyhelper.RL.helpers.replay_buffer import AER
from ludpyhelper.RL.Deep.tb import ModifiedTensorBoard
from tensorflow import keras

class DQ:
    def __init__(self, creat_model_func, log_dir, update_target_every, initial_memory_size, max_memory_size,
                 n_old, k,  epsilon=0.5, epsilon_update=None,
                 learning_rate=0.1, discount_factor=0.95,
                 shrinking_threshold=None, adaptively=True):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = self._init_q_table()
        self.epsilon = epsilon
        self.episode = 0
        self.epsilon_update = epsilon_update
        self.aer = AER(initial_memory_size, max_memory_size, n_old, k, shrinking_threshold, adaptively)

        self.learning_model = creat_model_func()
        self.target_model = creat_model_func()
        self.target_model.set_weights(self.learning_model.get_weights())

        self.tensorboard = ModifiedTensorBoard(log_dir=log_dir)
        self.update_target_every = update_target_every

        self.target_update_counter = 0

    def _random_action(self):
        action = np.random.randint(0, self.action_space)
        return action

    def cal_q_values(self, state, model="learning"):
        rstate = state.reshape(-1, *state.shape)
        if model == "target":
            q_values = self.target_model.predict(rstate)
        else:
            q_values = self.learning_model.predict(rstate)
        return q_values

    def act(self, state):
        q_values = self.cal_q_values(state)
        if np.random.random_sample() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = self._random_action()
        return action

    def cal_target(self, future_qs, reward):
        return reward + self.discount_factor * np.max(future_qs)

    def update_memory(self, current_state, action, new_state, reward, terminal_state):
        current_q = self.cal_q_values(current_state)[0][action]
        new_qs = self.cal_q_values(new_state, "target")[0]

        td = self.cal_target(new_qs, reward) - current_q

        self.aer.append([current_state, action, new_state, reward, terminal_state], td)

    def _update_epsilon(self):
        if self.epsilon_update is not None:
            if isinstance(self.epsilon_update, Iterable):
                if len(self.epsilon_update) == 3:
                    if self.epsilon_update[0] <= self.episode <= self.epsilon_update[1]:
                        self.epsilon -= self.epsilon_update[2]
                        if self.epsilon < 0:
                            self.epsilon = 0
            if isinstance(self.epsilon_update, Callable):
                self.epsilon = self.epsilon_update(self.epsilon, self.episode)

    def train_on_memory(self, batch_size, fit_batch_size, terminal_state, flat_sample=False, epsilon_update=True):
        batch = self.aer.sample(batch_size, flat_sample=flat_sample)

        current_states = [el[0] for el in batch]
        new_states = [el[2] for el in batch]

        current_states_qs = self.learning_model.predict(current_states)
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

        callbacks = [self.tensorboard] if terminal_state else None

        self.learning_model.fit(batch_x, batch_y, batch_size=fit_batch_size,
                                verbose=0, shuffle=False, callbacks=callbacks)

        if terminal_state:
            self.target_update_counter += 1
            if epsilon_update:
                self.episode += 1
                self.epsilon_update()

        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.learning_model.get_weights())
            self.target_update_counter = 0
