import rclpy
from rclpy.node import Node
import numpy as np
import random
from .rl_environment import UR3RLEnvironment

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_discrete_state(self, state):
        # print(state)
        return tuple(np.digitize(state, bins=np.linspace(-np.pi, np.pi, 10)))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(6))
        return np.argmax(self.q_table.get(state, np.zeros(6)))

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, np.zeros(6))[action]
        next_q = max(self.q_table.get(next_state, np.zeros(6)))
        self.q_table.setdefault(state, np.zeros(6))[action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def train(self, episodes=10):
        total_reward = 0
        for episode in range(episodes):
            state = self.get_discrete_state(self.env.reset_environment())
            done = False
            while not done:
                action = self.choose_action(state)
                self.env.set_action(action)
                next_state = self.get_discrete_state(self.env.get_obs())
                reward = self.env.compute_reward()
                total_reward += reward
                done = self.env.is_done()
                self.update_q_value(state, action, reward, next_state)
                state = next_state
            self.env.get_logger().info(f"Episode {episode}: Total Reward: {total_reward}")



def main(args=None):
    rclpy.init(args=args)
    env = UR3RLEnvironment()
    agent = QLearningAgent(env)
    agent.train()

    # rclpy.spin(node)
    env.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    
