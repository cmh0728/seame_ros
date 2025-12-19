# SEAME-ROS ADS

SEA:ME Team  autonomous driving stack built on ROS 2 and C++17 & Python. The stack is organized around a classic perception → planning → control pipeline to keep responsibilities clear and extensible.



## Software requirements
- ROS 2 humble
- python3.10.x
- c++17
- numpy<2
- open CV > 4.5 

## Workspace Layout

```
seame-ros/
├── README.md
└── src/
    ├── control/         
    ├── planning/         
    ├── perception/       
    ├── Params/      
    └── msg/   
```

Each package is a standard `ament_cmake` ROS 2 package with isolated responsibilities and shared interfaces. 

## Quick Start

1. Source your ROS 2 setup 
   ```bash
   source /opt/ros/humble/setup.zsh
   ```
2. Build the workspace from the repository root:
   ```bash
   colcon build --symlink-install
   ```
3. Source the workspace overlay:
   ```bash
   source install/setup.bash
   ```
4. Launch the full pipeline:
   ```bash
   ros2 launch main.launch
   ```

The demo publishes synthetic perception measurements, turns them into planning outputs, and finally generates normalized throttle/brake/steering commands.



## References
1. OpenCV: OpenCV is a popular open-source computer vision library that provides a wide range of tools and algorithms for image and video processing. Participants could use OpenCV for pre-processing the video footage, extracting features, and identifying the lane markings. Link: https://docs.opencv.org/4.5.4/d7/dbd/group__imgproc.html

2. TensorFlow: TensorFlow is an open-source machine learning framework that provides a wide range of tools for training deep neural networks. Participants could use TensorFlow for training a deep neural network for identifying extracted lane markings. Link: https://www.tensorflow.org/


3. ROS (Robot Operating System) (http://www.ros.org/): ROS is an open-source robotic operating system that provides a wide range of tools and libraries for developing autonomous systems, including robots and drones. It can be used to build the software framework for the Mail Delivering PiRacer and manage its various components, such as the navigation and delivery modules.

