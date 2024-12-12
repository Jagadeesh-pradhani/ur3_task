import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from .forward_kinematics import UR3ForwardKinematics
import time
from std_msgs.msg import Float64

class UR3RLEnvironment(Node):
    def __init__(self):
        super().__init__('ur3_rl_environment')

        # Joint names for UR3
        self.joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint', 
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]

        self.ACTIONS = [
            ('shoulder_pan_joint', 0.1),    # +0.1 for shoulder pan
            ('shoulder_pan_joint', -0.1),   # -0.1 for shoulder pan
            ('shoulder_lift_joint', 0.1),   # +0.1 for shoulder lift
            ('shoulder_lift_joint', -0.1),  # -0.1 for shoulder lift
            ('elbow_joint', 0.1),           # +0.1 for elbow
            ('elbow_joint', -0.1)           # -0.1 for elbow
        ]

        
        # Publisher for joint trajectory
        self.joint_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.episode_pub = self.create_publisher(Float64, '/ur3/qlearn/episode', 10)
        self.reward_pub = self.create_publisher(Float64, '/ur3/qlearn/reward', 10)

        # Subscriber for joint states
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Current joint positions
        self.current_joint_positions = UR3ForwardKinematics.inverse_kinematics(0.03, -0.385, 0.03)
        self.joint_positions_received = False
        
        # Initialize other RL variables
        self.goal_position = np.array([0.03, -0.385, 0.03]) 
        self.step_size = 0.1
        self.tolerance = 0.02
        self.max_steps = 100
        self.steps = 0
        
    def joint_state_callback(self, msg):
        """
        Callback function to update the joint positions when received.
        """
        self.current_joint_positions = msg.position[:6]  # Assuming first 6 joints
        self.joint_positions_received = True

    def set_action(self, action):
        """
        Perform an action that updates the joint positions and increments the step counter.
        """
        print(action)
        # joint_idx = action // 2  # Select joint based on action
        # direction = 1 if action % 2 == 0 else -1  # Determine direction of movement
        # self.current_joint_positions[joint_idx] += direction * self.step_size  # Update joint position
        
        joint_name, joint_delta = self.ACTIONS[action]
        joint_index = self.joint_names.index(joint_name)
        self.current_joint_positions[joint_index] += joint_delta

        # Publish the updated joint positions using the move_joints method
        self.move_joints(self.current_joint_positions)
        
        # Increment the step count
        self.steps += 1

    def get_obs(self):
        """
        Return the current observation (joint positions).
        """
        return np.array(self.current_joint_positions)

    def is_done(self):
        """
        Check if the episode is done based on the goal condition or maximum step count.
        """
        # Compute the position of the end effector using forward kinematics
        effector_pos = UR3ForwardKinematics.compute_forward_kinematics(self.current_joint_positions)[:3, 3]
        distance = np.linalg.norm(effector_pos - self.goal_position)
        print(distance, self.steps)
        return distance < self.tolerance or self.steps >= self.max_steps

    def compute_reward(self):
        """
        Compute the reward based on the current state and whether the goal is reached.
        """
        # Calculate distance to goal
        effector_pos = UR3ForwardKinematics.compute_forward_kinematics(self.current_joint_positions)[:3, 3]
        distance = np.linalg.norm(effector_pos - self.goal_position)
        return 100 if distance < self.tolerance else -1

    def reset_environment(self):
        """
        Reset the environment by returning the robot to its initial pose and resetting step count.
        """
        self.current_joint_positions = UR3ForwardKinematics.inverse_kinematics(0.03, -0.385, 0.03)
        self.steps = 0  # Reset step counter
        self.move_joints(self.current_joint_positions)  # Publish the reset positions
        print("Environment reset to initial positions.")
        return self.get_obs()  # Return the initial state (observation)

    def move_joints(self, joint_movements):
        """
        Move joints by the specified movements, considering delay.
        This function publishes the joint trajectory.
        """
        # Wait for initial joint positions to be received
        while not self.joint_positions_received:
            rclpy.spin_once(self)
        
        # Create a JointTrajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names  # Define joint names
        
        # Create the trajectory point with the updated joint positions
        point = JointTrajectoryPoint()
        point.positions = joint_movements  # Set the joint positions
        point.time_from_start.sec = 2  # Time to complete the movement (e.g., 2 seconds)
        
        # Append the trajectory point to the trajectory
        trajectory.points.append(point)
        
        # Publish the trajectory to the controller
        self.joint_pub.publish(trajectory)
        
        # Wait a little bit to ensure the movement is complete
        # rclpy.spin_once(self, timeout_sec=3)
        # time.sleep(3)
        
        # Print joint information after moving
        # self.print_joint_info(joint_movements)

    def print_joint_info(self, joint_positions):
        """
        Print the joint positions and the end effector information.
        """
        print("\n--- Joint Movement Report ---")
        
        # Print the joint positions
        for name, position in zip(self.joint_names, joint_positions):
            print(f"{name}: {position:.4f} radians")
        
        # Compute and print end effector position
        end_effector_transform = UR3ForwardKinematics.compute_forward_kinematics(joint_positions)
        print("\n--- End Effector Position ---")
        print(f"Position (x, y, z): ({end_effector_transform[0, 3]:.4f}, "
              f"{end_effector_transform[1, 3]:.4f}, {end_effector_transform[2, 3]:.4f}) meters")
        
        # Extract rotation matrix and compute Euler angles (roll, pitch, yaw)
        rotation_matrix = end_effector_transform[:3, :3]
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0],
                           np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        print("\n--- End Effector Orientation ---")
        print(f"Roll: {roll:.4f} radians")
        print(f"Pitch: {pitch:.4f} radians")
        print(f"Yaw: {yaw:.4f} radians")

def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS2 system
    node = UR3RLEnvironment()  # Create the RL environment node
    rclpy.spin(node)  # Keep the node running
    node.destroy_node()  # Clean up the node when finished
    rclpy.shutdown()  # Shutdown the ROS2 system

if __name__ == '__main__':
    main()
