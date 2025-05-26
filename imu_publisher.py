import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math
import numpy as np
import time


roll = 0
pitch = 0
yaw = 0

ang = [0.0,3.0,5.0]
lin = [0.0,2.0,4.0]

switch = 0

def euler_to_quaternion(roll, pitch, yaw):
    # Convert degrees to radians
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    
    x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return x, y, z, w

class IMUPublisher(Node):
    def __init__(self):
        super().__init__('imu_publisher')
        self.publisher_ = self.create_publisher(Imu, 'IMU_Publisher_test', 10)
        timer_period = 0.1  # seconds (10 Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        global roll, pitch, yaw, switch, ang, lin

        #change the pitch, roll and yaw values
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

        #change the angular velocity and linear acceleration values
        for i in range(3):
            if ang[i] < 10:
                ang[i] = ang[i] + 1.0
            else:
                ang[i] = 0.0
            if lin[i] < 10:
                lin[i] = lin[i] + 1.0
            else:
                lin[i] = 0.0
        
        
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
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing IMU: [{x:.3f}, {y:.3f}, {z:.3f}, {w:.3f}]')

def main(args=None):
    rclpy.init(args=args)
    node = IMUPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()