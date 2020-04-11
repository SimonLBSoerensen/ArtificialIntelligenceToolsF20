import numpy as np
from collections import defaultdict
from collections.abc import Iterable, Callable
from ludpyhelper.mics import load_from_pickle, save_to_pickle
from ludpyhelper.RL.helpers.replay_buffer import AER

class QTable:
    def __init__(self, action_space, initial_memory_size, max_memory_size,
                 n_old, k, epsilon=0.5, epsilon_update=None,
                 learning_rate=0.1, discount_factor=0.95, q_init=0,
                 shrinking_threshold=None, adaptively=True):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.q_init = q_init
        self.q_table = self._init_q_table()
        self.epsilon = epsilon
        self.episode = 0
        self.epsilon_update = epsilon_update
        self.aer = AER(initial_memory_size, max_memory_size, n_old, k, shrinking_threshold, adaptively)

    def _handle_init_q(self):
        if isinstance(self.q_init, Iterable):
            def init_q():
                return np.random.uniform(self.q_init[0], self.q_init[1], self.action_space)
        elif not isinstance(self.q_init, Callable):
            def init_q():
                return np.full(self.action_space, self.q_init, dtype=np.float32)
        else:
            init_q = self.q_init
        return init_q

    def _init_q_table(self):
        init_q = self._handle_init_q()
        return defaultdict(init_q)

    def _random_action(self):
        action = np.random.randint(0, self.action_space)
        return action

    def act(self, state):
        q_values = self.q_table[state]
        if np.random.random_sample() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = self._random_action()
        return action

    def cal_td(self, old_value, max_new, reward):
        td = reward + self.discount_factor * max_new - old_value
        return td

    def cal_update_value(self, old_value, max_new, reward):
        td = self.cal_td(old_value, max_new, reward)
        new_value = old_value + self.learning_rate * td
        return new_value, td

    def update(self, current_state, action, new_state, reward, terminal_state, save_experience=True):
        old_value = self.q_table[current_state][action]
        new_max = np.max(self.q_table[new_state])

        if not terminal_state:
            new_value, td = self.cal_update_value(old_value, new_max, reward)
        else:
            new_value = reward
            td = new_value - old_value
            if save_experience:
                self.episode += 1
                self._update_epsilon()

        if save_experience:
            self.aer.append([current_state, action, new_state, reward, terminal_state], td)
        self.q_table[current_state][action] = new_value

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

    def load_table(self, filename):
        q_table_dict = load_from_pickle(filename)
        init_q = self._handle_init_q()
        self.q_table = defaultdict(init_q, q_table_dict)

    def save_table(self, filename):
        q_dict = dict(self.q_table)
        save_to_pickle(filename, q_dict)

    def train_on_memory(self, batch_size, flat_sample=False):
        batch = self.aer.sample(batch_size, flat_sample=flat_sample)

        for current_state, action, new_state, reward, terminal_state in batch:
            self.update(current_state, action, new_state, reward, terminal_state, save_experience=False)



class NQTable(QTable):
    def __init__(self, action_space, n_q_tabels, initial_memory_size, max_memory_size,
                 n_old, k, epsilon=0.5, epsilon_update=None,
                 learning_rate=0.1, discount_factor=0.95, q_init=0,
                 shrinking_threshold=None, adaptively=True):
        self.n_q_tabels = n_q_tabels
        super().__init__(action_space, initial_memory_size, max_memory_size,
                 n_old, k, epsilon, epsilon_update,
                 learning_rate, discount_factor, q_init,
                 shrinking_threshold, adaptively)
        self._init_q_table()

    def _init_q_table(self):
        init_q = self._handle_init_q()
        self.q_table = [defaultdict(init_q) for _ in range(self.n_q_tabels)]

    def update(self, current_state, action, new_state, reward, terminal_state, save_experience=True):
        if self.n_q_tabels > 1:
            update_table, estimate_table = np.random.choice(np.arange(self.n_q_tabels), 2, replace=False)
        else:
            update_table, estimate_table = (0, 0)

        max_action = np.argmax(self.q_table[update_table][new_state])
        new_max = self.q_table[estimate_table][new_state][max_action]
        old_value = self.q_table[update_table][current_state][action]

        if not terminal_state:
            new_value, td = self.cal_update_value(old_value, new_max, reward)
        else:
            new_value = reward
            td = new_value - old_value
            if save_experience:
                self.episode += 1
                self._update_epsilon()

        if save_experience:
            self.aer.append([current_state, action, new_state, reward, terminal_state], td)

        self.q_table[update_table][current_state][action] = new_value

    def act(self, state):
        if np.random.random_sample() > self.epsilon:
            q_values = [qt[state] for qt in self.q_table]
            if len(q_values) > 1:
                q_sums = np.array(q_values).sum(axis=0)
            else:
                q_sums = q_values[0]
            action = np.argmax(q_sums)
        else:
            action = self._random_action()
        return action

    def save_table(self, filename):
        q_dicts = [dict(table) for table in self.q_table]
        save_to_pickle(filename, q_dicts)

    def load_table(self, filename):
        q_dicts = load_from_pickle(filename)
        init_q = self._handle_init_q()
        self.q_table = [defaultdict(init_q, q_dict) for q_dict in q_dicts]


#Clipped Double Q-learning