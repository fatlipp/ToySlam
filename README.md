## Simple SLAM Implementation
This project is a basic implementation of SLAM using a simulated 2D LiDAR. The goal is to demonstrate the fundamental principles of SLAM, including robot pose estimation and optimization using virtual measurements.

![SLAM](/assets/SLAM.png)

### Features
- 2D LiDAR Simulation: A virtual LiDAR sensor is simulated, providing distance measurements to surrounding objects. This data is used to map the environment and estimate the robot's pose.
 - Robot Pose Optimization: The robot's pose (position and orientation) is optimized using virtual measurements, refining its estimated location over time.

### Requirements
 - numpy: for matrix operations

 
### Key concepts
 In this implementation, SLAM is used to optimize the robot's pose without constructing a map. The key aspects are:

 - State Representation: The robot's state is represented by 3d matrix:
**[R t]**
**[0 1]**

 - Odometry:
**[R t]**
**[0 1]**

 - Graph2d: consists of robot poses in 2d and relative landmarks observations

 - Although the current implementation is focused on 2D, the robot's state is represented in 3D to facilitate future developments, such as extending the system to 3D SLAM.

### Further development
 - Building a map and navigation
 - Using global LMs
 - BA
 - Performance optimization