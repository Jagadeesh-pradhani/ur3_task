import rclpy
from rclpy.node import Node
import random
import numpy as np
import time
import math
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# Custom environment to replace openai_ros
class CustomUR3Env:
    def __init__(self, max_episode_steps):
        self.node = rclpy.create_node('custom_ur3_env')
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Subscribe to /joint_states
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publisher for commands
        self.trajectory_pub = self.node.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.current_joint_states = None

    def joint_state_callback(self, msg):
        self.current_joint_states = msg

    def reset(self):
        self.current_step = 0
        # Reset the robot to an initial position
        self.publish_trajectory([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        while self.current_joint_states is None:
            rclpy.spin_once(self.node)
        return list(self.current_joint_states.position)

    def step(self, action):
        self.current_step += 1

        # Apply action (e.g., move joints)
        target_positions = [a for a in action]
        self.publish_trajectory(target_positions)

        # Wait for state update
        rclpy.spin_once(self.node)
        observation = list(self.current_joint_states.position)

        reward = self.calculate_reward(observation)
        done = self.current_step >= self.max_episode_steps
        return observation, reward, done, {}

    def calculate_reward(self, state):
        # Define a reward function based on state
        return -np.linalg.norm(np.array(state) - np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

    def publish_trajectory(self, positions):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rclpy.time.Duration(seconds=2.0).to_msg()
        traj_msg.points = [point]
        self.trajectory_pub.publish(traj_msg)

    def close(self):
        self.node.destroy_node()


# Define the Q-learning class
class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            q = [q[i] + random.random() * mag - 0.5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)

# ROS2 Node for Q-learning
class QLearningNode(Node):
    def __init__(self):
        super().__init__('ur3_gym_learn')
        self.get_logger().info("Start!")

        # Load parameters from ROS2 parameter server
        self.declare_parameter("/ur3/alpha", 0.1)
        self.declare_parameter("/ur3/epsilon", 0.1)
        self.declare_parameter("/ur3/gamma", 0.9)
        self.declare_parameter("/ur3/epsilon_discount", 0.99)
        self.declare_parameter("/ur3/nepisodes", 1000)
        self.declare_parameter("/ur3/nsteps", 100)
        
        self.alpha = self.get_parameter("/ur3/alpha").value
        self.epsilon = self.get_parameter("/ur3/epsilon").value
        self.gamma = self.get_parameter("/ur3/gamma").value
        self.epsilon_discount = self.get_parameter("/ur3/epsilon_discount").value
        self.nepisodes = self.get_parameter("/ur3/nepisodes").value
        self.nsteps = self.get_parameter("/ur3/nsteps").value

        # Create the custom environment
        self.env = CustomUR3Env(max_episode_steps=self.nsteps)
        self.get_logger().info("UR3 environment done")
        self.get_logger().info("Starting Learning")

        # Initialize Q-learning
        self.qlearn = QLearn(actions=range(self.env.action_space.n), alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon)
        self.initial_epsilon = self.qlearn.epsilon

        # Start training loop
        self.train_agent()

    def train_agent(self):
        start_time = time.time()
        highest_reward = 0
        self.get_logger().info("Q Learn Initialized")

        for episode in range(self.nepisodes):
            self.get_logger().info(f"EPISODE=> {episode}")
            cumulated_reward = 0
            done = False

            # Decaying epsilon for exploration vs exploitation
            if self.qlearn.epsilon > 0.05:
                self.qlearn.epsilon *= self.epsilon_discount

            # Initialize the environment and get first state of the robot
            observation = self.env.reset()
            self.get_logger().info("Environment Reset")
            state = '-'.join(map(str, observation))

            # Loop through steps for the current episode
            for step in range(self.nsteps - 1):
                exit_flag = False
                self.get_logger().info(f"Start-> Episode: {episode}, Step: {step}")

                # Pick an action based on the current state
                action = self.qlearn.chooseAction(state)

                # Execute the action in the environment and get feedback
                observation, reward, done, info = self.env.step(action)
                next_state = '-'.join(map(str, observation))
                self.get_logger().info(f"Observation: {observation}, Reward: {reward}")

                cumulated_reward += reward
                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward

                # Make the algorithm learn based on the results
                self.get_logger().info(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

                # Q-learning update rule
                self.qlearn.learn(state, action, reward, next_state)

                if done:
                    self.get_logger().info("Done!")
                    break

                state = next_state  # Update state for next iteration

            # Log episode statistics
            elapsed_time = time.time() - start_time
            self.get_logger().info(f"Episode {episode + 1}: Reward = {cumulated_reward}, Time: {elapsed_time:.2f} seconds")

        self.get_logger().info(f"Training Completed. Best reward: {highest_reward}")

        # Close the environment
        self.env.close()

def main(args=None):
    rclpy.init(args=args)
    qlearning_node = QLearningNode()
    rclpy.spin(qlearning_node)
    qlearning_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()