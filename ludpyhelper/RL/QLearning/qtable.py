import numpy as np
from collections import defaultdict
from collections.abc import Iterable, Callable
from ludpyhelper.mics import load_from_pickle, save_to_pickle
from ludpyhelper.RL.helpers.replay_buffer import AER
from threading import Thread, Lock

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
        self.mutex = Lock()
        self.update_count = 0

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
        self.mutex.acquire()
        q_values = self.q_table[state]
        self.mutex.release()
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

    def update(self, current_state, action, new_state, reward, terminal_state, save_experience=True, new_state_list=False, auto_update_episode=True):
        self.update_count += 1

        self.mutex.acquire()
        old_value = self.q_table[current_state][action]
        new_max = np.max(self.q_table[new_state])
        self.mutex.release()

        if new_state_list:
            over_all_max = None

            for n_state in new_state:
                self.mutex.acquire()
                new_max = np.max(self.q_table[n_state])
                self.mutex.release()

                if over_all_max is None:
                    over_all_max = new_max
                else:
                    over_all_max = max(over_all_max, new_max)

        if not terminal_state:
            new_value, td = self.cal_update_value(old_value, new_max, reward)
        else:
            new_value = reward
            td = new_value - old_value
            if save_experience and auto_update_episode:
                self.update_episode()

        if save_experience:
            self.aer.append([current_state, action, new_state, reward, terminal_state, new_state_list], td)

        self.mutex.acquire()
        self.q_table[current_state][action] = new_value
        self.mutex.release()

    def update_episode(self):
        self.episode += 1
        self._update_epsilon()


    def _update_epsilon(self):
        if self.epsilon_update is not None:
            self.epsilon = self.epsilon_update(self.episode)

    def load_table(self, filename):
        q_table_dict = load_from_pickle(filename)
        init_q = self._handle_init_q()
        self.q_table = defaultdict(init_q, q_table_dict)

    def save_table(self, filename):
        q_dict = dict(self.q_table)
        save_to_pickle(filename, q_dict)

    def train_on_memory(self, batch_size, flat_sample=False):
        self.mutex.acquire()
        batch = self.aer.sample(batch_size, flat_sample=flat_sample)
        self.mutex.release()

        for current_state, action, new_state, reward, terminal_state, new_state_list in batch:
            self.update(current_state, action, new_state, reward, terminal_state, save_experience=False, new_state_list=new_state_list)



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

    def update(self, current_state, action, new_state, reward, terminal_state, save_experience=True, new_state_list=False, auto_update_episode=True):
        self.update_count += 1

        if self.n_q_tabels > 1:
            update_table, estimate_table = np.random.choice(np.arange(self.n_q_tabels), 2, replace=False)
        else:
            update_table, estimate_table = (0, 0)

        if new_state_list:
            over_all_max = None

            for n_state in new_state:
                self.mutex.acquire()
                max_action = np.argmax(self.q_table[update_table][n_state])
                new_max = self.q_table[estimate_table][n_state][max_action]
                self.mutex.release()

                if over_all_max is None:
                    over_all_max = new_max
                else:
                    over_all_max = max(over_all_max, new_max)
        else:
            self.mutex.acquire()
            max_action = np.argmax(self.q_table[update_table][new_state])
            new_max = self.q_table[estimate_table][new_state][max_action]
            self.mutex.release()

        self.mutex.acquire()
        old_value = self.q_table[update_table][current_state][action]
        self.mutex.release()

        if not terminal_state:
            new_value, td = self.cal_update_value(old_value, new_max, reward)
        else:
            new_value = reward
            td = new_value - old_value
            if save_experience and auto_update_episode:
                self.update_episode()

        if save_experience:
            self.aer.append([current_state, action, new_state, reward, terminal_state, new_state_list], td)

        self.mutex.acquire()
        self.q_table[update_table][current_state][action] = new_value
        self.mutex.release()

    def act(self, state):
        if np.random.random_sample() > self.epsilon:
            self.mutex.acquire()
            q_values = [qt[state] for qt in self.q_table]
            self.mutex.release()

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