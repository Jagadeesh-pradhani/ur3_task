import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import time

class UR3HarmonicMotion(Node):
    def __init__(self):
        super().__init__('ur3_harmonic_motion')

        # Publisher for joint commands
        self.joint_command_pub = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)

        # Joint state variables for six joints
        self.current_positions = [0.0] * 6  # Default initial positions for all joints

        self.start_time = time.time()
        self.timer = self.create_timer(0.5, self.update_joints)  # Send commands at 0.5s intervals

    def update_joints(self):
        t = time.time() - self.start_time

        # Calculate joint positions based on equations for the first three joints
        shoulder_pan_pos = (math.pi / 2) * (1 - math.cos(0.2 * math.pi * t))
        shoulder_lift_pos = (math.pi / 4) * (math.cos(0.4 * math.pi * t) - 1)
        elbow_pos = (math.pi / 2) * math.sin(0.2 * math.pi * t)

        # Keep other joints stationary
        wrist_1_pos = 0.0
        wrist_2_pos = 0.0
        wrist_3_pos = 0.0

        # Create and publish joint command for all six joints
        joint_command = Float64MultiArray()
        joint_command.data = [
            shoulder_pan_pos, 
            shoulder_lift_pos, 
            elbow_pos, 
            wrist_1_pos, 
            wrist_2_pos, 
            wrist_3_pos
        ]
        self.joint_command_pub.publish(joint_command)

        # Log current positions
        self.get_logger().info(f"Time: {t:.2f}s")
        self.get_logger().info(f"Shoulder Pan: {shoulder_pan_pos:.2f} rad")
        self.get_logger().info(f"Shoulder Lift: {shoulder_lift_pos:.2f} rad")
        self.get_logger().info(f"Elbow: {elbow_pos:.2f} rad")


def main(args=None):
    rclpy.init(args=args)
    node = UR3HarmonicMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
