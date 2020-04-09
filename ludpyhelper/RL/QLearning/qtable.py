import numpy as np
from collections import defaultdict
from collections.abc import Iterable, Callable


class QTable:
    def __init__(self, action_space, epsilon=0.5, epsilon_update=None,  learning_rate=0.1, discount_factor=0.95, q_init=0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.q_init = q_init
        self.q_table = self.__init_q_table()
        self.epsilon = epsilon
        self.episode = 0
        self.epsilon_update = epsilon_update

    def __init_q_table(self):
        if isinstance(self.q_init, Iterable):
            def init_q():
                return np.random.uniform(self.q_init[0], self.q_init[1], self.action_space)
        elif not isinstance(self.q_init, Callable):
            def init_q():
                return np.full(self.action_space, self.q_init, dtype=np.float32)
        else:
            init_q = self.q_init

        return defaultdict(init_q)

    def act(self, state):
        q_values = self.q_table[state]
        if np.random.random_sample() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, len(q_values))
        return action

    def cal_update_value(self, old_value, max_new, reward):
        return old_value + self.learning_rate * (reward + self.discount_factor * max_new - old_value)

    def update(self, current_state, action, new_state, reward, terminal_state):
        old_value = self.q_table[current_state][action]
        new_max = np.max(self.q_table[new_state])

        if not terminal_state:
            new_value = self.cal_update_value(old_value, new_max, reward)
        else:
            new_value = reward
            self.episode += 1
            self.__update_epsilon()

        self.q_table[current_state][action] = new_value
        pass

    def __update_epsilon(self):
        if self.epsilon_update is not None:
            if isinstance(self.epsilon_update, Iterable):
                if len(self.epsilon_update) == 3:
                    if self.epsilon_update[0] <= self.episode <= self.epsilon_update[1]:
                        self.epsilon -= self.epsilon_update[2]
            if isinstance(self.epsilon_update, Callable):
                self.epsilon = self.epsilon_update(self.epsilon, self.episode)
