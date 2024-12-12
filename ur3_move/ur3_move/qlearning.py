import numpy as np

class QLearning:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros(state_space + (len(action_space),))  # Initialize Q-table
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.action_space))
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action])
