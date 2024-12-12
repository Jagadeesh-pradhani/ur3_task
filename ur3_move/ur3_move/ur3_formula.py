import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
from time import time

class UR3ForwardPositionOscillator(Node):
    def __init__(self):
        super().__init__('ur3_forward_position_oscillator')

        # Joint names for UR3
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint'
        ]

        # Publisher for forward position controller
        self.position_publisher = self.create_publisher(
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

        # Timer for periodic joint movement
        self.create_timer(0.1, self.move_joints)
        self.start_time = time()

    def joint_state_callback(self, msg):
        # Update current joint positions when received
        self.current_positions = msg.position
        self.joint_positions_received = True

    def move_joints(self):
        # Wait for initial joint positions to be received
        if not self.joint_positions_received:
            self.get_logger().warn('Waiting for initial joint states...')
            return

        # Calculate elapsed time
        elapsed_time = time() - self.start_time

        # Calculate new joint positions
        new_positions = [
            np.pi/2 * (1 - np.cos(0.2 * np.pi * elapsed_time)),  # Shoulder Pan
            np.pi/4 * (np.cos(0.4 * np.pi * elapsed_time) - 1),  # Shoulder Lift
            np.pi/2 * np.sin(0.2 * np.pi * elapsed_time)  # Elbow
        ]
        print(new_positions)

        # Create Float64MultiArray message
        joint_commands = Float64MultiArray()
        joint_commands.data = new_positions

        # Publish joint commands
        self.position_publisher.publish(joint_commands)

        # Log joint movement information
        self.log_joint_info(new_positions)

    def log_joint_info(self, new_positions):
        self.get_logger().info('\n--- Joint Movement Report ---')
        for i, (name, position) in enumerate(zip(self.joint_names, new_positions)):
            self.get_logger().info(f"{name}: {position:.4f} radians")

def main(args=None):
    rclpy.init(args=args)
    node = UR3ForwardPositionOscillator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('UR3 Forward Position Oscillator node stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()