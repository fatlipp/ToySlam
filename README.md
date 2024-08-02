## Simple SLAM Implementation
This project is a basic implementation of SLAM using a simulated 2D LiDAR. The goal is to demonstrate the fundamental principles of SLAM, including robot pose estimation and optimization using graph.

![SLAM](/assets/SLAM.png)

### Features
- 2D LiDAR Simulation: A virtual LiDAR sensor is simulated, providing distance measurements to surrounding objects. This data is used to map the environment and estimate the robot's pose.
 - Robot Pose Optimization: The robot's pose (position and orientation) is optimized using noisy landmarks and noisy robot positions.

### Requirements
 - numpy: for matrix operations

### Key concepts
 - State Representation: The robot's state is represented by 3x3 matrix:
**[R t]**
**[0 1]**

 - Odometry:
**[R t]**
**[0 1]**

 - Graph2d: consists of robot poses in 2d and relative landmarks observations
 - OptGraph: consists of robot poses in 2d and relative landmarks observations, that used for optimization

### Simplification
Each point in the environment has its own ID, which is used to match landmarks

### Further development
 - Virtual measurements and Loop closure
 - Using a map for navigation
 - BA
 - Performance optimization