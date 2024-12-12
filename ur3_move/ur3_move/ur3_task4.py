import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import math

class UR3TrajectoryActionClient(Node):
    def __init__(self):
        super().__init__('ur3_trajectory_action_client')

        # Action client for the joint trajectory controller
        self.action_client = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')

        # Joint names for the UR3 robot
        self.joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint', 
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]

    def send_trajectory(self):
        # Wait for the action server to become available
        self.get_logger().info('Waiting for action server...')
        self.action_client.wait_for_server()

        # Create a goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        # Generate trajectory points for 10 seconds
        duration = 10.0
        time_step = 0.1  # Smaller time step for smoother trajectory
        t = 0.0

        while t <= duration:
            shoulder_pan_pos = (math.pi / 2) * (1 - math.cos(0.2 * math.pi * t))
            shoulder_lift_pos = (math.pi / 4) * (math.cos(0.4 * math.pi * t) - 1)
            elbow_pos = (math.pi / 2) * math.sin(0.2 * math.pi * t)

            # Keep wrist joints stationary
            wrist_1_pos = 0.0
            wrist_2_pos = 0.0
            wrist_3_pos = 0.0

            point = JointTrajectoryPoint()
            point.positions = [
                shoulder_pan_pos, 
                shoulder_lift_pos, 
                elbow_pos, 
                wrist_1_pos, 
                wrist_2_pos, 
                wrist_3_pos
            ]
            point.time_from_start = rclpy.duration.Duration(seconds=t).to_msg()
            # Log the trajectory point
            # self.get_logger().info(f"Generated point at t={t}: {point.positions}")
            goal_msg.trajectory.points.append(point)
            t += time_step

        # Send the goal
        self.get_logger().info('Sending trajectory to action server...')
        send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server.')
            return

        self.get_logger().info('Goal accepted by action server.')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        # Extract feedback information
        desired_positions = feedback_msg.feedback.desired.positions
        actual_positions = feedback_msg.feedback.actual.positions
        error_positions = feedback_msg.feedback.error.positions
        
        desired_velocities = feedback_msg.feedback.desired.velocities
        actual_velocities = feedback_msg.feedback.actual.velocities
        
        # Prepare the feedback in a human-readable format
        feedback_str = "Feedback received:\n"
        feedback_str += "Time: {}\n".format(feedback_msg.feedback.header.stamp)
        
        feedback_str += "\nDesired Positions: {}\n".format(desired_positions)
        feedback_str += "Actual Positions: {}\n".format(actual_positions)
        feedback_str += "Position Errors: {}\n".format(error_positions)
        
        feedback_str += "\nDesired Velocities: {}\n".format(desired_velocities)
        feedback_str += "Actual Velocities: {}\n".format(actual_velocities)
        
        # You can also print velocities, accelerations, and other fields as needed
        self.get_logger().info(feedback_str)

    def result_callback(self, future):
        result = future.result().result
        if result.error_code != 0:
            self.get_logger().error(f'Result received: {result.error_code}, {result.error_string}')
        else:
            self.get_logger().info('Trajectory executed successfully.')

        # Stop spinning after processing the result
        # rclpy.get_default_context().shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = UR3TrajectoryActionClient()
    try:
        node.send_trajectory()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()