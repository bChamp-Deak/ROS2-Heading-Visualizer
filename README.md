# ROS2-Heading-Visualizer
This is a program is a python script using PyQT6 to visualize a heading message from a ROS2 IMU or odom topic.

First install the dependencies:

```
pip install PyQt6
pip install pyqtgraph
```

Download and run heading_visualizer.py.

You can manually type in either Euler Angles or Quaternion angles and they will be converted automatically.

You can also do this live on a ROS2 topic by:
  - Select a topic from the Select topic dropdown box
  - Press Start Stream. The Quaternion and Eular angles will updated as they are received
  - If you want to see more information about the topic, press Show Extrat Fields button.

An example program is included that will publish an IMU topic that sweaps its values through each axis. To run this, download imu_publisher.py and run it.
