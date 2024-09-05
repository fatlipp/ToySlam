## Simple SLAM Implementation
This project is a basic implementation of SLAM using a simulated 2D LiDAR. The goal is to demonstrate the fundamental principles of SLAM, including robot pose estimation and optimization using graph.

![SLAM](/assets/SLAM.png)

### Features
- 2D LiDAR Simulation: A virtual LiDAR sensor is simulated, providing distance measurements to surrounding objects. This data is used to map the environment and estimate the robot's pose.
 - Robot Pose Optimization: The robot's pose (position and orientation) is optimized using noisy landmarks and noisy robot positions.
 - Remote CPP optimizer:
   - 1. Build `cpp` folder:
     Example of using conan:
        * conan install . --output-folder=./build -s compiler.cppstd=gnu20 -s compiler.version=13 --build=missing
        * cmake --preset conan-release -DWITH_CUDA=OFF [ON/OFF]
        * cmake --build --preset conan-release
   - 2. Run: graph_optimizer HOST PORT ITERATIONS PIPELINE SOLVER (./bin/graph_optimizer "127.0.0.1" "8888" "50" cpu eigen)
        - ITERATIONS - [int value >= 1]
        - PIPELINE: [cpu/gpu]
        - SOLVER: [cuda/eigen] (if `PIPELINE == GPU` => `SOLVER = CUDA`)
   - 3. Run `python3 python/slam_main.py`

### Requirements
 - numpy: for matrix operations

 - CPP (optional, for remote optimization):
    * Eigen - for matrix operations
    * Conan 2 (optional) - packet manager
    * CUDA (optional)

### Pipeline
There is 4 different pipelines:
 - Python UI + Python optimizer
 - Python UI + CPP optimizer
 - Python UI + CPP optimizer (CUDA matrix solver)
 - Python UI + Full CUDA optimizer

### Key concepts
 - State Representation: The robot's state is represented by 3x3 matrix:
**[R t]**
**[0 1]**

 - Odometry:
**[R t]**
**[0 1]**

 - Graph2d: consists of robot poses in 2d and relative landmarks observations
 - OptGraph: consists of robot poses in 2d and relative landmarks observations, that used for optimization
 - 2 types of edges: [ODOM, LM]

### Simplification
Each point in the environment has its own ID, which is used to match landmarks

### Further development
 - Adding More Edge and Vertex types (2d, 3d, BA, Virtual Meas.)
 - Using a map for navigation
 - Performance optimization
 - CUDA opt improvements