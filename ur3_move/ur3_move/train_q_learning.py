import rclpy
from rclpy.node import Node
import numpy as np
import random
import matplotlib.pyplot as plt
from .rl_environment import UR3RLEnvironment
from std_msgs.msg import Float64
import argparse

class QLearningAgent:
    def __init__(self,
                 env, 
                 alpha=0.1,
                 gamma=0.9,    
                 epsilon=0.3,   
                 epsilon_decay=0.99, 
                 episodes=500):

        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.episodes = episodes
        self.q_table = {}

        # Data for plotting
        self.episode_rewards = []

    def get_discrete_state(self, state):
        """
        Convert continuous state to a discretized state tuple.
        This method uses fixed bins to reduce the continuous state space.
        """
        # Define bins for each joint angle
        bins = [
            np.linspace(-np.pi, np.pi, 20),  # shoulder pan
            np.linspace(-np.pi, np.pi, 20),  # shoulder lift
            np.linspace(-np.pi, np.pi, 20),  # elbow
            np.linspace(-np.pi, np.pi, 20),  # wrist 1
            np.linspace(-np.pi, np.pi, 20),  # wrist 2
            np.linspace(-np.pi, np.pi, 20)   # wrist 3
        ]
        
        # Digitize each state dimension
        discrete_state = tuple(np.digitize(s, bins=b) for s, b in zip(state, bins))
        return discrete_state

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        """
        # Exploration
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.env.ACTIONS) - 1)
        
        # Exploitation
        state_q_values = self.q_table.get(state, np.zeros(len(self.env.ACTIONS)))
        return np.argmax(state_q_values)

    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-values using the Q-learning update rule.
        """
        # Initialize Q-values for state if not exists
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.env.ACTIONS))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.env.ACTIONS))
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def train(self):
        """
        Train the Q-learning agent.
        """
        val = Float64()
        for episode in range(self.episodes):
            # Reset environment and get initial state
            initial_obs = self.env.reset_environment()
            state = self.get_discrete_state(initial_obs)
            
            total_reward = 0
            done = False
            
            while not done:
                # Choose and execute action
                action = self.choose_action(state)
                self.env.set_action(action)
                
                # Get next observation and reward
                next_obs = self.env.get_obs()
                reward = self.env.compute_reward()
                next_state = self.get_discrete_state(next_obs)
                
                # Update Q-values
                self.update_q_value(state, action, reward, next_state)
                
                total_reward += reward
                state = next_state
                done = self.env.is_done()
                val.data = float(reward)
                self.env.reward_pub.publish(val)
            
            # Decay exploration rate
            self.epsilon *= self.epsilon_decay
            
            # Store total reward for plotting
            self.episode_rewards.append(total_reward)
            
            val.data = float(episode + 1)
            self.env.episode_pub.publish(val)

            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {self.epsilon:.4f}")

    def plot_training_results(self):
        """
        Plot the results of training: Episode vs Total Reward.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, label='Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Performance')
        plt.legend()
        plt.grid()
        plt.show()

def parse_arguments():
    """
    Parse command-line arguments for Q-learning hyperparameters.
    """
    parser = argparse.ArgumentParser(description='Q-Learning Agent Configuration')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Discount factor (default: 0.9)')
    parser.add_argument('--epsilon', type=float, default=0.3,
                        help='Initial exploration rate (default: 0.3)')
    parser.add_argument('--epsilon-decay', type=float, default=0.99,
                        help='Exploration rate decay (default: 0.99)')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes (default: 500)')
    
    return parser.parse_args()

def main(args=None):
    cli_args = parse_arguments()
    rclpy.init(args=args)
    env = UR3RLEnvironment()
    agent = QLearningAgent(
        env,
        alpha=cli_args.alpha,
        gamma=cli_args.gamma,
        epsilon=cli_args.epsilon,
        epsilon_decay=cli_args.epsilon_decay,
        episodes=cli_args.episodes
        )
    agent.train()
    agent.plot_training_results()
    env.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()