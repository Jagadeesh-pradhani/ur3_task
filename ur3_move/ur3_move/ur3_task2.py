#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import threading
import time

class UR3Controller(Node):
    def __init__(self):
        super().__init__('ur3_controller')

        self.joint_state = JointState()
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # Publisher for joint commands
        self.joint_command_publisher = self.create_publisher(Float64MultiArray, "/forward_position_controller/commands", 10)

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

        # Transform listener for TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Thread management
        self.running = True
        self.timer_thread = threading.Thread(target=self.timer_task)
        self.input_thread = threading.Thread(target=self.input_task)
        self.timer_thread.start()
        self.input_thread.start()

    def joint_state_callback(self, msg):
        self.joint_state = msg

    def move_joint(self, joint_index, position):
        target_joint = self.joint_names[joint_index]

        # Create and publish command for all joints, updating only the target joint
        command = Float64MultiArray()
        current_positions = [0.0] * len(self.joint_names)

        if self.joint_state.position:
            current_positions = list(self.joint_state.position[:len(self.joint_names)])

        current_positions[joint_index] = position
        command.data = current_positions

        self.joint_command_publisher.publish(command)

        self.get_logger().info(f"Moving {target_joint} to position {position}...")
        time.sleep(1.0)

        try:
            joint_index_in_state = self.joint_state.name.index(target_joint)
            actual_position = self.joint_state.position[joint_index_in_state]
            error = position - actual_position

            transform = self.tf_buffer.lookup_transform(
                'base_link', 'wrist_3_link', rclpy.time.Time()
            )

            self.get_logger().info(f"Actual position of {target_joint}: {actual_position:.10f}")
            self.get_logger().info(f"Error in position of {target_joint}: {error:.10f}")

            translation = transform.transform.translation
            rotation = transform.transform.rotation

            self.get_logger().info("End effector relative to base:")
            self.get_logger().info(f"Position: x:{translation.x}, y:{translation.y}, z:{translation.z}")
            self.get_logger().info(f"Orientation: x:{rotation.x}, y:{rotation.y}, z:{rotation.z}, w:{rotation.w}")

        except Exception as e:
            self.get_logger().error(f"Error getting transform or joint state: {e}")

    def timer_task(self):
        while rclpy.ok() and self.running:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 'wrist_3_link', rclpy.time.Time()
                )
                # self.get_logger().info("Periodic transform lookup succeeded.")
            except Exception as e:
                pass
                # self.get_logger().error(f"Periodic transform lookup failed: {e}")
            time.sleep(1.0)

    def input_task(self):
        try:
            while self.running and rclpy.ok():
                print("Select joints to move:")
                for idx, joint_name in enumerate(self.joint_names):
                    print(f"{idx + 1}. {joint_name}")
                print("7. Exit")

                selection = int(input("Select joint: "))
                if selection == 7:
                    self.running = False
                    rclpy.shutdown()
                    break
                else:
                    joint_index = selection - 1
                    position = float(input(f"What position to move {self.joint_names[joint_index]} to? "))
                    self.move_joint(joint_index, position)
        except KeyboardInterrupt:
            self.get_logger().info("Input task interrupted.")

    def stop_threads(self):
        self.running = False
        if self.timer_thread.is_alive():
            self.timer_thread.join()
        if self.input_thread.is_alive():
            self.input_thread.join()


def main(args=None):
    rclpy.init(args=args)
    controller = UR3Controller()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down...")

    finally:
        controller.stop_threads()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
