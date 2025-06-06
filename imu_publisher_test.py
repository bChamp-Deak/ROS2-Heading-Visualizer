##
# @file imu_publisher_test.py
# @brief This script publishes some fummy IMU data to be able to test the GUI program
# @author Benjamin
# @date 2025-06-06
##

import rclpy  # ROS2 Python client library
from rclpy.node import Node  # Base class for ROS2 nodes
from sensor_msgs.msg import Imu  # IMU message type
import math  # For trigonometric functions
import numpy as np  # For quaternion calculation
import time  # Not used, can be removed

# Initialize roll, pitch, yaw angles (in degrees)
roll = 0
pitch = 0
yaw = 0

# Initialize angular velocity and linear acceleration arrays
ang = [0.0, 3.0, 5.0]
lin = [0.0, 2.0, 4.0]

# Switch variable to control which angle is being incremented
switch = 0

"""
@brief Convert Euler angles (in degrees) to quaternion (x, y, z, w).
@param roll Roll angle in degrees.
@param pitch Pitch angle in degrees.
@param yaw Yaw angle in degrees.
@return Tuple (x, y, z, w) representing the quaternion.
"""
def euler_to_quaternion(roll, pitch, yaw):
    # Convert degrees to radians
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    
    # Quaternion conversion formula
    x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return x, y, z, w

"""
@class IMUPublisher
@brief ROS2 Node that publishes simulated IMU messages on the 'IMU_Publisher_test' topic.
"""
class IMUPublisher(Node):
    """
    @brief Constructor. Initializes the publisher and timer.
    """
    def __init__(self):
        super().__init__('imu_publisher')
        # Create a publisher for the Imu message type
        self.publisher_ = self.create_publisher(Imu, 'IMU_Publisher_test', 10)
        timer_period = 0.1  # Publish at 10 Hz
        # Create a timer to call timer_callback periodically
        self.timer = self.create_timer(timer_period, self.timer_callback)

    """
    @brief Called periodically to update and publish IMU data.
    Increments roll, pitch, or yaw, and cycles angular velocity and linear acceleration.
    Publishes the IMU message.
    """
    def timer_callback(self):
        global roll, pitch, yaw, switch, ang, lin

        # Increment roll, pitch, or yaw depending on the switch value
        if switch == 0:
            roll = roll + 5
            if roll >= 360:
                switch = switch + 1
                roll = 0
        if switch == 1:
            pitch = pitch + 5
            if pitch >= 360:
                switch = switch + 1
                pitch = 0
        if switch == 2:
            yaw = yaw + 5
            if yaw >= 360:
                switch = 0
                yaw = 0
                pitch = 0
                yaw = 0

        # Increment angular velocity and linear acceleration values cyclically
        for i in range(3):
            if ang[i] < 10:
                ang[i] = ang[i] + 1.0
            else:
                ang[i] = 0.0
            if lin[i] < 10:
                lin[i] = lin[i] + 1.0
            else:
                lin[i] = 0.0
        
        # Create and populate the IMU message
        msg = Imu()
        x, y, z, w = euler_to_quaternion(roll, pitch, yaw)
        msg.orientation.x = x
        msg.orientation.y = y
        msg.orientation.z = z
        msg.orientation.w = w
        msg._angular_velocity.x = ang[0]
        msg._angular_velocity.y = ang[1]
        msg._angular_velocity.z = ang[2]
        msg._linear_acceleration.x = lin[0]
        msg._linear_acceleration.y = lin[1]
        msg._linear_acceleration.z = lin[2]
        # Publish the message
        self.publisher_.publish(msg)
        # Log the published quaternion
        self.get_logger().info(f'Publishing IMU: [{x:.3f}, {y:.3f}, {z:.3f}, {w:.3f}]')

"""
@brief Main entry point for the ROS2 node.
@param args Optional command-line arguments.
Initializes ROS2, starts the IMUPublisher node, and spins.
"""
def main(args=None):
    rclpy.init(args=args)
    node = IMUPublisher()
    try:
        rclpy.spin(node)  # Keep the node running
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
