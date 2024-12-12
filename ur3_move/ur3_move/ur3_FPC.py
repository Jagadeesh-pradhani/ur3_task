import rclpy
from rclpy.node import Node
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
            
            # Compute transformation matrix for this joint
            Rz = np.array([
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle),  np.cos(angle), 0, 0],
                [0,              0,             1, 0],
                [0,              0,             0, 1]
            ])
            
            Tz = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, d],
                [0, 0, 0, 1]
            ])
            
            Tx = np.array([
                [1, 0, 0, a],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            Rx = np.array([
                [1, 0,               0,              0],
                [0, np.cos(alpha), -np.sin(alpha), 0],
                [0, np.sin(alpha),  np.cos(alpha), 0],
                [0, 0,               0,              1]
            ])
            
            # Update transformation
            T = T @ Rz @ Tz @ Tx @ Rx
        
        return T

class UR3JointController(Node):
    def __init__(self):
        super().__init__('ur3_joint_controller')
        
        # Joint names for UR3
        self.joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint', 
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]
        
        # Create forward position controller publisher
        self.forward_position_publisher = self.create_publisher(
            Float64MultiArray, 
            '/forward_position_controller/commands', 
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
        
        # Create a timer to periodically output diagnostic information
        # self.create_timer(5.0, self.diagnostics_callback)
        
    def joint_state_callback(self, msg):
        """Update current joint positions"""
        self.current_positions = list(msg.position)
        self.joint_positions_received = True
    
    def move_joints(self, joint_movements):
        """Move joints using forward position controller"""
        # Wait for initial joint positions to be received
        while not self.joint_positions_received:
            rclpy.spin_once(self)
        
        # Calculate new absolute positions
        new_positions = [
            curr + move 
            for curr, move in zip(self.current_positions, joint_movements)
        ]
        
        # Create Float64MultiArray message
        joint_commands = Float64MultiArray()
        joint_commands.data = new_positions
        
        # Publish joint commands
        self.forward_position_publisher.publish(joint_commands)
        rclpy.spin_once(self, timeout_sec=3)
        # Print movement information
        self.print_joint_info(new_positions, joint_movements)
    
    def print_joint_info(self, new_positions, movements):
        """Print detailed joint movement information"""
        self.get_logger().info("\n--- Joint Movement Report ---")
        
        # Print final positions and movements
        for i, (name, final, movement) in enumerate(zip(self.joint_names, new_positions, movements)):
            self.get_logger().info(f"{name}:")
            self.get_logger().info(f"  Final Position: {final:.4f} radians")
            self.get_logger().info(f"  Movement Amount: {movement:.4f} radians")
        
        # Compute end effector position
        try:
            end_effector_transform = UR3ForwardKinematics.compute_forward_kinematics(new_positions)
            
            # Log end effector position
            self.get_logger().info("\n--- End Effector Position ---")
            self.get_logger().info(f"Position (x, y, z): "
                  f"({end_effector_transform[0,3]:.4f}, "
                  f"{end_effector_transform[1,3]:.4f}, "
                  f"{end_effector_transform[2,3]:.4f}) meters")
            
            # Extract and log orientation
            rotation_matrix = end_effector_transform[:3,:3]
            roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            pitch = np.arctan2(-rotation_matrix[2,0], 
                               np.sqrt(rotation_matrix[2,1]**2 + rotation_matrix[2,2]**2))
            yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
            
            self.get_logger().info("\n--- End Effector Orientation ---")
            self.get_logger().info(f"Roll: {roll:.4f} radians")
            self.get_logger().info(f"Pitch: {pitch:.4f} radians")
            self.get_logger().info(f"Yaw: {yaw:.4f} radians")
        
        except Exception as e:
            self.get_logger().error(f"Error computing end effector position: {e}")
    
    def diagnostics_callback(self):
        """Periodic diagnostics callback"""
        # Log current joint states and status
        if self.joint_positions_received:
            self.get_logger().info("Diagnostic Update:")
            self.get_logger().info(f"Current Joint Positions: {[f'{p:.4f}' for p in self.current_positions]}")

def main(args=None):
    rclpy.init(args=args)
    node = UR3JointController()
    
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
            
            # Spin to process callbacks
            # rclpy.spin_once(node)
            
    except KeyboardInterrupt:
        print("\nNode terminated by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
