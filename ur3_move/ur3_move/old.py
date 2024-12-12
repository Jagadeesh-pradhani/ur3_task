#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
import time

class UR3Move(Node):
    def __init__(self):
        super().__init__('ur3_move')
        
        # Publishers and Subscribers
        self.trajectory_publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        # TF listener for end-effector pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Current joint positions
        self.current_positions = [0.0] * 6

    def joint_state_callback(self, msg):
        """Update the current joint positions."""
        self.current_positions = list(msg.position)

    def get_end_effector_pose(self):
        """Get the pose of the end effector using TF with retry logic."""
        retries = 0
        while retries < 10:  # Retry for a maximum of 10 attempts
            try:
                # time.sleep(0.5)
                # Wait for the transform to be available
                self.tf_buffer.can_transform('base_link', 'wrist_3_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1))

                # Look up the transform between 'base_link' and 'wrist_3_link'
                transform = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())

                # Print the translation and rotation
                self.get_logger().info(f"Translation: {transform.transform.translation}")
                self.get_logger().info(f"Rotation (Quaternion): {transform.transform.rotation}")

                # Convert Quaternion to Roll-Pitch-Yaw
                from tf_transformations import euler_from_quaternion
                rpy = euler_from_quaternion([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ])

                # Print the RPY in radians and degrees
                self.get_logger().info(f"Rotation (RPY in radians): {rpy}")
                self.get_logger().info(f"Rotation (RPY in degrees): {[r * 180.0 / 3.141592 for r in rpy]}")

                return transform.transform.translation, transform.transform.rotation

            except Exception as e:
                self.get_logger().warn(f"Could not get transform: {str(e)}")
                retries += 1
                time.sleep(1)  # Wait for 1 second before retrying

        self.get_logger().error("Failed to get transform after 10 retries.")
        return None, None


    def move_joints(self):
        """Prompt user for joint movements and publish trajectory commands."""
        while rclpy.ok():
            self.get_logger().info("Current joint positions: " + str(self.current_positions))
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                                          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

            point = JointTrajectoryPoint()
            point.positions = []

            # Get user input for each joint
            for i, current_pos in enumerate(self.current_positions):
                try:
                    move_by = float(input(f"Enter relative movement for Joint {i + 1} (radians): "))
                    point.positions.append(current_pos + move_by)
                except ValueError:
                    self.get_logger().error("Invalid input! Please enter a number.")
                    return

            point.time_from_start.sec = 2  # Set execution time
            trajectory_msg.points = [point]

            # Publish trajectory
            self.trajectory_publisher.publish(trajectory_msg)
            self.get_logger().info(f"Published trajectory: {point.positions}")

            # Print end-effector pose
            position, orientation = self.get_end_effector_pose()
            if position and orientation:
                self.get_logger().info(f"End Effector Position: {position}")
                self.get_logger().info(f"End Effector Orientation: {orientation}")


def main(args=None):
    rclpy.init(args=args)
    ur3_move = UR3Move()
    ur3_move.move_joints()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
