import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np

class UR3ForwardKinematics:
    # UR3 DH Parameters (in meters and radians)
    # Order: [a, d, alpha, theta]
    DH_PARAMS = [
        [0,       0.1519,  np.pi/2, 0],  # Base to Shoulder
        [0,       0,      0,        0],  # Shoulder to Elbow
        [0.24365, 0,      0,        0],  # Elbow to Wrist 1
        [0.21325, 0.1125, np.pi,    0],  # Wrist 1 to Wrist 2
        [0,       0.0825, -np.pi/2, 0],  # Wrist 2 to Wrist 3
        [0,       0.0815, 0,        0]   # Wrist 3 to End Effector
    ]

    @staticmethod
    def rotation_matrix_z(theta):
        """Rotation matrix around Z-axis"""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [0,              0,             1, 0],
            [0,              0,             0, 1]
        ])

    @staticmethod
    def translation_matrix(x, y, z):
        """Translation matrix"""
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def rotation_matrix_x(alpha):
        """Rotation matrix around X-axis"""
        return np.array([
            [1, 0,               0,              0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha),  np.cos(alpha), 0],
            [0, 0,               0,              1]
        ])

    @classmethod
    def compute_forward_kinematics(cls, joint_angles):
        """
        Compute end effector position and orientation using forward kinematics
        
        :param joint_angles: List of 6 joint angles in radians
        :return: 4x4 transformation matrix representing end effector pose
        """
        # Start with identity matrix
        T = np.eye(4)
        
        # Compute transformation for each joint
        for i, (angle, params) in enumerate(zip(joint_angles, cls.DH_PARAMS)):
            # Unpack DH parameters
            a, d, alpha, _ = params
            
            # Z-axis rotation by joint angle
            Rz = cls.rotation_matrix_z(angle)
            
            # Translation along Z
            Tz = cls.translation_matrix(0, 0, d)
            
            # Translation along X
            Tx = cls.translation_matrix(a, 0, 0)
            
            # X-axis rotation
            Rx = cls.rotation_matrix_x(alpha)
            
            # Combine transformations
            T = T @ Rz @ Tz @ Tx @ Rx
        
        return T


class RLTrainingEnv(Node):
    def __init__(self):
        super().__init__('rl_training_env')

        # Publishers and Subscribers
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Robot State
        self.current_positions = [0.0] * 3  # Assume [shoulder_pan, shoulder_lift, elbow]
        self.joint_positions_received = False

        # RL Parameters
        self.goal_position = np.array([0.03, 0.385, 0.03])  # (x, y, z)
        self.tolerance = 0.02
        self.step_size = 0.1
        self.max_steps = 100

        self.joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint'
        ]

    def joint_state_callback(self, msg):
        self.current_positions = msg.position[:3]  # Assuming the first 3 joints are used
        self.joint_positions_received = True

    def set_action(self, action):
        """
        Perform the specified action (increment/decrement joint positions).
        :param action: List of joint movements for the first 3 joints [delta_pan, delta_lift, delta_elbow]
        """
        while not self.joint_positions_received:
            rclpy.spin_once(self)

        # Calculate new positions for all six joints
        new_positions = [
            curr + act if i < len(action) else curr  # Update first 3 joints; keep others static
            for i, (curr, act) in enumerate(zip(self.current_positions, action + [0.0, 0.0, 0.0]))
        ]

        # Create JointTrajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # point = JointTrajectoryPoint()
        # point.positions = new_positions
        # point.time_from_start = rclpy.time.Duration(seconds=2.0)
        # trajectory.points = [point]
        
        point = JointTrajectoryPoint()
        point.positions = new_positions
        point.time_from_start.sec = 2 

        trajectory.points.append(point)

        # Publish the trajectory
        self.get_logger().info(f"Publishing trajectory: {trajectory}")
        self.trajectory_pub.publish(trajectory)
        self.get_logger().info("Trajectory published")


    def get_obs(self):
        """
        Return the current joint positions.
        """
        return self.current_positions

    def is_done(self):
        """
        Check if the episode is complete.
        """
        end_effector_pos = self.compute_end_effector_position()
        dist_to_goal = np.linalg.norm(end_effector_pos - self.goal_position)
        return dist_to_goal < self.tolerance

    def compute_reward(self):
        """
        Compute the reward for the current state.
        """
        end_effector_pos = self.compute_end_effector_position()
        dist_to_goal = np.linalg.norm(end_effector_pos - self.goal_position)
        if dist_to_goal < self.tolerance:
            return 100  # Goal reached
        return -1  # Step penalty

    def reset_environment(self):
        """
        Reset the robot to the initial pose (all joints at 0 radians).
        """
        self.set_action([-pos for pos in self.current_positions])  # Reset to 0, 0, 0
        return self.get_obs()

    def compute_end_effector_position(self):
        """
        Use forward kinematics to compute the end effector position.
        """
        fk = UR3ForwardKinematics.compute_forward_kinematics(self.current_positions)
        return fk[:3, 3]

def main():
    rclpy.init()
    env = RLTrainingEnv()

    # Example training loop
    for episode in range(10):
        
        env.reset_environment()
        for step in range(env.max_steps):
            action = np.random.choice([-env.step_size, env.step_size], size=3)  # Random action
            env.set_action(action)
            obs = env.get_obs()
            reward = env.compute_reward()
            if env.is_done():
                break

    rclpy.shutdown()

if __name__ == '__main__':
    main()