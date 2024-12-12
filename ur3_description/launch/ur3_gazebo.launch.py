from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessStart
from launch.actions import RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    

    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "controllers_file",
            default_value="ur3_controllers.yaml",
            description="controller",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "runtime_config_package",
            default_value="ur3_description",
            description='Package with the controller\'s configuration in "config" folder. \
        Usually the argument is not set, it enables use of a custom setup.',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value='""',
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )
    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="joint_trajectory_controller",
            description="Robot controller to start.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "start_joint_controller",
            default_value="true",
            description="Enable headless mode for robot control",
        )
    )

    # Initialize Arguments
    prefix = LaunchConfiguration("prefix")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    controllers_file = LaunchConfiguration("controllers_file")
    runtime_config_package = LaunchConfiguration("runtime_config_package")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")
    start_joint_controller = LaunchConfiguration("start_joint_controller")

    initial_joint_controllers = PathJoinSubstitution(
        [FindPackageShare(runtime_config_package), "config", controllers_file]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])]
        ),
        launch_arguments={"gz_args": " -r -v 3 empty.sdf --render-engine ogre"}.items(),
    )

    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("ur3_description"), "urdf", "ur3.urdf.xacro"]
            ),
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
            " ",
            "name:=",
            "ur3",
            " ",
            "prefix:=",
            prefix,
            " ",
            "simulation_controllers:=",
            initial_joint_controllers,
        ]
    )

    robot_description = {"robot_description": robot_description_content}

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True}, robot_description],
    )
    
    pedestal = PathJoinSubstitution(
        [FindPackageShare("ur3_description"), "models", "pedestal.urdf"]
    )    
    spawn_pedestal = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", "pedestal", "-file", pedestal],
        output="screen",
    )

    table = PathJoinSubstitution(
        [FindPackageShare("ur3_description"), "models", "table.urdf"]
    ) 
    spawn_table = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", "table", "-file", table, "-y", "0.5"],
        output="screen",
    )

    cube = PathJoinSubstitution(
        [FindPackageShare("ur3_description"), "models", "cube.sdf"]
    ) 
    spawn_cube = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", "cube", "-file", cube, "-x", "0.03", "-y", "0.385", "-z", "0.3"],
        output="screen",
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", "robot_description", "-name", "ur3_system_position", "-z", "0.3"],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",],
    )

    initial_joint_controller_spawner_started = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[initial_joint_controller, "-c", "/controller_manager"],
        condition=IfCondition(start_joint_controller),
    )
    
    initial_joint_controller_spawner_stopped = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[initial_joint_controller, "-c", "/controller_manager", "--stopped"],
        condition=UnlessCondition(start_joint_controller),
    )



    nodes = [
        gazebo,
        robot_state_publisher_node,
        joint_state_broadcaster_spawner,
        spawn_pedestal,
        spawn_table,
        spawn_cube,
        spawn_robot,
        initial_joint_controller_spawner_stopped,
        initial_joint_controller_spawner_started,

        

    ]

    return LaunchDescription(declared_arguments + nodes)
