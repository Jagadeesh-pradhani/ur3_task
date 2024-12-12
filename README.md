# UR3 Robot Task Repository

## Project Overview
This repository contains a ROS 2 Jazzy-based project for controlling and training a UR3 robot in a Gazebo simulation environment. The project includes multiple tasks demonstrating robot control, movement, and reinforcement learning techniques.

## Prerequisites

### System Requirements
- Ubuntu 22.04 LTS
- ROS 2 Jazzy
- Gazebo Harmonic

### Dependencies

Install the following dependencies:
```bash
sudo apt install ros-jazzy-ros2-control \
                 ros-jazzy-ros2-controllers \
                 ros-jazzy-joint-state-publisher-gui \
                 ros-jazzy-ros-gz-sim \
                 ros-jazzy-controller-manager \
                 ros-jazzy-gz-ros2-control \
                 git
```

## Gazebo Installation

### Install Gazebo Harmonic
```bash
sudo apt-get update
sudo apt-get install curl lsb-release gnupg

sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic
```

## Workspace Setup

### Clone the Repository
```bash
mkdir -p ~/ws/src/
cd ~/ws/src/
git clone https://github.com/Jagadeesh-pradhani/ur3_task.git -b tasks
```

### Install Dependencies
```bash
cd ~/ws/
rosdep update
rosdep install -y --from-paths src --ignore-src -r
```

### Build the Workspace
```bash
colcon build
source ~/ws/install/setup.bash
```

## Tasks

### Task 1: Run UR3 Robot with Controller
```bash
# Terminal 1
cd ~/ws/
source ~/ws/install/setup.bash
ros2 launch ur3_description ur3_gazebo.launch.py initial_joint_controller:=forward_position_controller
```

### Task 2: Forward Position Controller
```bash
# Terminal 1: Launch Gazebo
cd ~/ws/
source ~/ws/install/setup.bash
ros2 launch ur3_description ur3_gazebo.launch.py initial_joint_controller:=forward_position_controller

# Terminal 2: Move Robot
cd ~/ws/
source ~/ws/install/setup.bash
ros2 run ur3_move ur3_task2
```
- Select a joint and specify its position in radians

### Task 3: Oscillation Movement
```bash
# Terminal 1: Launch Gazebo
cd ~/ws/
source ~/ws/install/setup.bash
ros2 launch ur3_description ur3_gazebo.launch.py initial_joint_controller:=forward_position_controller

# Terminal 2: Run Oscillation
cd ~/ws/
source ~/ws/install/setup.bash
ros2 run ur3_move ur3_task3
```

### Task 4: Action Client Joint Trajectory
```bash
# Terminal 1: Launch Gazebo
cd ~/ws/
source ~/ws/install/setup.bash
ros2 launch ur3_description ur3_gazebo.launch.py

# Terminal 2: Run Action Client
cd ~/ws/
source ~/ws/install/setup.bash
ros2 run ur3_move ur3_task4
```

### Task 5: Q-Learning Training
```bash
# Terminal 1: Launch Gazebo
cd ~/ws/
source ~/ws/install/setup.bash
ros2 launch ur3_description ur3_gazebo.launch.py

# Terminal 2: Train Robot
cd ~/ws/
source ~/ws/install/setup.bash
ros2 run ur3_move ur3_task5 --episodes 1000
```
- Trains the robot for specified number of episodes
- Generates a reward plot at the end of training
- ![image](https://github.com/user-attachments/assets/7b6a2cce-dda5-497d-a123-87351848a563)


## Troubleshooting

### Gazebo Harmonic Not Detected
If Gazebo Harmonic is not detected, use:
```bash
ros2 launch ur3_description ur3_gazebo.launch.py gz_version:=8
```

## Additional Notes
- Press `Ctrl+C` to close terminals
- In Task 2, select `7` to exit the movement interface

