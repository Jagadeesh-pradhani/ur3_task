import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
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

class UR3JointMover(Node):
    def __init__(self):
        super().__init__('ur3_joint_mover')
        
        # Joint names for UR3
        self.joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint', 
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]
        
        # Publishers for joint trajectory and state
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        # Subscriber to get current joint states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Current joint positions
        self.current_positions = [0.0] * len(self.joint_names)
        self.joint_positions_received = False
        
    def joint_state_callback(self, msg):
        # Update current joint positions when received
        self.current_positions = msg.position
        self.joint_positions_received = True
    
    def move_joints(self, joint_movements):
        # Wait for initial joint positions to be received
        while not self.joint_positions_received:
            rclpy.spin_once(self)
        
        # Calculate new absolute positions
        new_positions = [
            curr + move 
            for curr, move in zip(self.current_positions, joint_movements)
        ]
        
        # Create joint trajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = new_positions
        point.time_from_start.sec = 2  # 2-second movement
        
        trajectory.points.append(point)
        
        # Publish trajectory
        self.trajectory_publisher.publish(trajectory)
        
        # Wait a bit to ensure movement is complete
        rclpy.spin_once(self, timeout_sec=3)
        
        # Print joint information
        self.print_joint_info(new_positions, joint_movements)
    
    def print_joint_info(self, new_positions, movements):
        print("\n--- Joint Movement Report ---")
        
        # Print final positions and errors
        for i, (name, final, movement) in enumerate(zip(self.joint_names, new_positions, movements)):
            print(f"{name}:")
            print(f"  Final Position: {final:.4f} radians")
            print(f"  Movement Amount: {movement:.4f} radians")
        
        # Compute and print end effector position
        end_effector_transform = UR3ForwardKinematics.compute_forward_kinematics(new_positions)
        
        print("\n--- End Effector Position ---")
        print(f"Position (x, y, z): ({end_effector_transform[0,3]:.4f}, "
              f"{end_effector_transform[1,3]:.4f}, {end_effector_transform[2,3]:.4f}) meters")
        
        # Extract orientation (rotation matrix)
        rotation_matrix = end_effector_transform[:3,:3]
        
        # Convert rotation matrix to roll, pitch, yaw (Euler angles)
        roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        pitch = np.arctan2(-rotation_matrix[2,0], 
                           np.sqrt(rotation_matrix[2,1]**2 + rotation_matrix[2,2]**2))
        yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        
        print("\n--- End Effector Orientation ---")
        print(f"Roll: {roll:.4f} radians")
        print(f"Pitch: {pitch:.4f} radians")
        print(f"Yaw: {yaw:.4f} radians")

def main(args=None):
    rclpy.init(args=args)
    node = UR3JointMover()
    
    try:
        while rclpy.ok():
            print("\n--- Joint Movement Interface ---")
            joint_movements = [0.0] * len(node.joint_names)
            
            # Prompt for movement of each joint
            for i, name in enumerate(node.joint_names):
                try:
                    movement = float(input(f"Enter amount to move {name} by (radians): "))
                    joint_movements[i] = movement
                except ValueError:
                    print("Invalid input. Using 0 for this joint.")
                    joint_movements[i] = 0.0
            
            # Move joints
            node.move_joints(joint_movements)
            
    except KeyboardInterrupt:
        print("\nNode terminated by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()