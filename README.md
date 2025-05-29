# ROS2-Heading-Visualizer
This is a program is a python script using PyQT6 to visualize a heading message from a ROS2 IMU or odom topic.

First, install the dependencies:

```
pip install PyQt6
pip install pyqtgraph
```

Download and run heading_visualizer.py.

You can manually type in either Euler Angles or Quaternion angles and they will be converted automatically.

You can also do this live on a ROS2 topic by:
  - Select a topic from the Select topic dropdown box
  - Optionally select a frame to transform the topic into. By default, no frame will be used
  - Press Start Stream. The Quaternion and Eular angles will be updated as they are received
  - If you want to see more information about the topic, press Show Extra Fields button.
