##
# @file heading_visualizer.py
# @brief Quaternion/Euler Angle Visualizer with ROS2 IMU integration and 3D visualization.
#
# This script provides a PyQt6 GUI for visualizing and converting between quaternion and Euler angles,
# subscribing to ROS2 IMU topics, and displaying IMU data in real time. It features a 3D visualization
# of orientation and extra IMU fields such as linear acceleration and angular velocity.
#
# @author Benjamin
# @date 2025-05-26
##

import sys
import math
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry  # Add import for Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_ros
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs  # <-- Add this line
from builtin_interfaces.msg import Time as MsgTime
from rclpy.time import Time
import yaml

# Force message registration for TF2
import geometry_msgs.msg._pose_stamped

##
# @brief Convert quaternion to Euler angles (roll, pitch, yaw).
# @param q Quaternion as [x, y, z, w].
# @return List of Euler angles [roll, pitch, yaw] in degrees.
def quaternion_to_euler(q):
    # Unpack quaternion
    x, y, z, w = q

    # Calculate roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    
    # Calculate pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2  # Clamp value
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    
    # Calculate yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    # Convert radians to degrees
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

##
# @brief Convert Euler angles to quaternion.
# @param e Euler angles [roll, pitch, yaw] in degrees.
# @return Quaternion as (x, y, z, w).
def euler_to_quaternion(e):
    # Convert degrees to radians
    roll, pitch, yaw = [math.radians(a) for a in e]
    
    # Compute quaternion components
    x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return x, y, z, w

##
# @brief Rotate a 3D vector by a quaternion.
# @param v Vector [x, y, z].
# @param q Quaternion [x, y, z, w].
# @return Rotated vector [x, y, z].
def rotate_vector_by_quaternion(v, q):
    # Quaternion multiplication: q * v * q_conj
    x, y, z = v
    qx, qy, qz, qw = q
    # Quaternion representing the vector
    vx, vy, vz, vw = x, y, z, 0
    # First multiplication: q * v
    rx = qw * vx + qy * vz - qz * vy
    ry = qw * vy + qz * vx - qx * vz
    rz = qw * vz + qx * vy - qy * vx
    rw = -qx * vx - qy * vy - qz * vz
    # Second multiplication: result * q_conj
    cx = -qx
    cy = -qy
    cz = -qz
    cw = qw
    fx = rx * cw + rw * cx + ry * cz - rz * cy
    fy = ry * cw + rw * cy + rz * cx - rx * cz
    fz = rz * cw + rw * cz + rx * cy - ry * cx
    return [fx, fy, fz]

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    Args:
        q1: [x1, y1, z1, w1]
        q2: [x2, y2, z2, w2]
    Returns:
        [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return [x, y, z, w]

##
# @class ROS2IMUFinder
# @brief ROS2 node for discovering IMU and Odometry topics.
class ROS2IMUFinder(Node):
    ##
    # @brief Constructor for ROS2IMUFinder.
    def __init__(self):
        super().__init__('imu_finder')
        self.topics = []

    ##
    # @brief Find all available IMU and Odometry topics.
    # @return List of tuples (topic_name, msg_type).
    def find_imu_topics(self):
        self.topics = []
        topic_list = self.get_topic_names_and_types()
        # Loop through all topics and check their types
        for name, types in topic_list:
            # If topic is IMU, add to list
            if 'sensor_msgs/msg/Imu' in types:
                self.topics.append((name, 'Imu'))
            # If topic is Odometry, add to list
            elif 'nav_msgs/msg/Odometry' in types:
                self.topics.append((name, 'Odometry'))
        return self.topics

##
# @class RefreshingComboBox
# @brief QComboBox that refreshes its items when opened.
class RefreshingComboBox(QComboBox):
    ##
    # @brief Constructor for RefreshingComboBox.
    # @param refresh_callback Function to call before showing popup.
    def __init__(self, refresh_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refresh_callback = refresh_callback

    ##
    # @brief Override to refresh items before showing popup.
    def showPopup(self):
        # Refresh items before showing the popup
        self.refresh_callback()
        super().showPopup()

##
# @class QuaternionEulerWidget
# @brief Main widget for quaternion/Euler visualization and IMU streaming.
class QuaternionEulerWidget(QWidget):
    ##
    # @brief Constructor for QuaternionEulerWidget.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quaternion/Euler Angle Visualizer")
        self.resize(800, 600)
        self.imu_sub = None
        self.extra_fields_visible = False
        self.extra_fields_widgets = []
        self.selected_frame = None
        self.tf_buffer = None
        self.tf_listener = None
        self.init_ui()
        self.init_ros()
        self.update_graph()

        # Timer to spin ROS2 node
        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.spin_ros)
        self.ros_timer.start(50)  # 20 Hz

    ##
    # @brief Initialize the user interface.
    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Euler and Quaternion input layout ---
        angle_layout = QHBoxLayout()

        # Euler angles group
        euler_group = QGroupBox("Euler Angles (deg)")
        euler_layout = QVBoxLayout()
        self.euler_labels = []
        self.euler_edits = []
        # Create input fields for Roll, Pitch, Yaw
        for name in ['Roll', 'Pitch', 'Yaw']:
            # Add label and input for each Euler angle
            h = QHBoxLayout()
            lbl = QLabel(name)
            edit = QLineEdit("0.0")
            edit.setReadOnly(False)  # Editable at startup
            edit.editingFinished.connect(self.on_euler_changed)  # Update quaternion when Euler is changed
            h.addWidget(lbl)
            h.addWidget(edit)
            euler_layout.addLayout(h)
            self.euler_labels.append(lbl)
            self.euler_edits.append(edit)
        euler_group.setLayout(euler_layout)
        angle_layout.addWidget(euler_group)

        # Quaternion group
        quat_group = QGroupBox("Quaternion (x, y, z, w)")
        quat_layout = QVBoxLayout()
        self.quat_labels = []
        self.quat_edits = []
        # Create input fields for x, y, z, w
        for name in ['x', 'y', 'z', 'w']:
            # Add label and input for each quaternion component
            h = QHBoxLayout()
            lbl = QLabel(name)
            edit = QLineEdit("0.0")
            edit.editingFinished.connect(self.on_quat_changed)
            h.addWidget(lbl)
            h.addWidget(edit)
            quat_layout.addLayout(h)
            self.quat_labels.append(lbl)
            self.quat_edits.append(edit)
        quat_group.setLayout(quat_layout)
        angle_layout.addWidget(quat_group)

        main_layout.addLayout(angle_layout)

        # --- 3D Graph layout ---
        graph_layout = QHBoxLayout()
        self.graph_widget = gl.GLViewWidget()
        self.graph_widget.setCameraPosition(distance=5)
        graph_layout.addWidget(self.graph_widget, stretch=1)
        main_layout.addLayout(graph_layout)

        # --- Topic and frame selection, buttons ---
        bottom_layout = QHBoxLayout()
        self.combo = RefreshingComboBox(self.refresh_topics)
        self.combo.setEditable(False)
        self.combo.addItem("Select topic...")

        self.frame_combo = RefreshingComboBox(self.refresh_frames)
        self.frame_combo.setEditable(False)
        self.frame_combo.addItem("Select frame...")

        self.load_btn = QPushButton("Start Stream")
        self.expand_btn = QPushButton("Show Extra Fields")
        bottom_layout.addWidget(self.combo)
        bottom_layout.addWidget(self.frame_combo)
        bottom_layout.addWidget(self.load_btn)
        bottom_layout.addWidget(self.expand_btn)
        main_layout.addLayout(bottom_layout)

        # Connect buttons and combo boxes to handlers
        self.load_btn.clicked.connect(self.toggle_stream)
        self.expand_btn.clicked.connect(self.toggle_extra_fields)
        self.frame_combo.currentIndexChanged.connect(self.on_frame_selected)
        self.combo.activated.connect(self.on_topic_selected)  # Only triggers on actual selection

        # --- Extra fields area (hidden by default) ---
        self.extra_fields_group = QGroupBox("Other Values")
        self.extra_fields_layout = QVBoxLayout()
        self.extra_fields_group.setLayout(self.extra_fields_layout)
        self.extra_fields_group.setVisible(False)
        main_layout.addWidget(self.extra_fields_group)

        self.setLayout(main_layout)

    ##
    # @brief Handle topic selection change.
    # @param idx Index of the selected topic in the combo box.
    def on_topic_selected(self, idx):
        new_topic = self.combo.currentText()
        # Only act if a real topic is selected, not empty, and it's different from the current streaming topic
        if new_topic and new_topic != "Select topic..." and (
            not hasattr(self, "current_streaming_topic") or new_topic != getattr(self, "current_streaming_topic", None)
        ):
            # If streaming, stop previous stream and start new one
            if self.load_btn.text() == "Stop Stream":
                self.stop_stream()
                self.set_textboxes_editable(False)
                self.start_stream()
            self.current_streaming_topic = new_topic
        # If "Select topic..." is chosen, stop stream and clear current topic
        elif new_topic == "Select topic...":
            if self.load_btn.text() == "Stop Stream":
                self.stop_stream()
            self.current_streaming_topic = None

    ##
    # @brief Refresh the list of available frames.
    def refresh_frames(self):
        if self.tf_buffer is None:
            return
        self.frame_combo.clear()
        self.frame_combo.addItem("Select frame...")
        try:
            yaml_str = self.tf_buffer.all_frames_as_yaml()
            frames_dict = yaml.safe_load(yaml_str)
            frames_set = set()
            # Loop through all frames in the TF tree
            if isinstance(frames_dict, dict):
                for child, info in frames_dict.items():
                    # Add child frame
                    frames_set.add(child)
                    # Add parent frame if it exists
                    parent = info.get('parent', None)
                    if parent:
                        frames_set.add(parent)
            # Add all frames to the combo box
            for frame in sorted(frames_set):
                self.frame_combo.addItem(frame)
        except Exception as e:
            print(f"Could not get frames: {e}")

    ##
    # @brief Handle frame selection.
    # @param idx Index of the selected frame in the combo box.
    def on_frame_selected(self, idx):
        # If "Select frame..." is chosen, clear selection
        if self.frame_combo.currentText() == "Select frame...":
            self.selected_frame = None
        else:
            # Otherwise, set selected frame
            self.selected_frame = self.frame_combo.currentText()

    ##
    # @brief Initialize ROS2 node and TF2 listener.
    def init_ros(self):
        rclpy.init(args=None)
        self.ros_node = ROS2IMUFinder()
        try:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.ros_node)
        except Exception as e:
            print(f"TF2 initialization failed: {e}")
            self.tf_buffer = None
            self.tf_listener = None

    ##
    # @brief Toggle IMU stream on/off.
    def toggle_stream(self):
        # If not streaming, start stream and make text boxes read-only
        if self.load_btn.text() == "Start Stream":
            self.set_textboxes_editable(False)
            self.start_stream()
        # If streaming, stop stream and make text boxes editable
        else:
            self.set_textboxes_editable(True)
            self.stop_stream()

    ##
    # @brief Set editability of text boxes.
    # @param editable True to make editable, False to make read-only.
    def set_textboxes_editable(self, editable):
        # Set quaternion text boxes
        for edit in self.quat_edits:
            edit.setReadOnly(not editable)
        # Set Euler text boxes (usually read-only)
        for edit in self.euler_edits:
            edit.setReadOnly(not editable)
        # Set extra fields (always read-only)
        if hasattr(self, 'extra_fields_widgets'):
            for container in self.extra_fields_widgets:
                for child in container.findChildren(QLineEdit):
                    child.setReadOnly(True)

    ##
    # @brief Start streaming IMU or Odometry data from selected topic.
    def start_stream(self):
        topic = self.combo.currentText()
        if topic == "Select topic...":
            return

        if self.imu_sub is not None:
            self.ros_node.destroy_subscription(self.imu_sub)
            self.imu_sub = None

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        msg_type = self.topic_types.get(topic, 'Imu')

        # --- Transform functions ---

        def transform_pose_quat(position, orientation, from_frame, to_frame, stamp):
            """
            Transform a pose (position + orientation) from from_frame to to_frame.
            Used for Odometry messages.
            """
            if self.tf_buffer is None or to_frame is None or to_frame == "Select frame...":
                return position, orientation, True
            if not from_frame:
                print("No frame_id in message!")
                return position, orientation, False
            try:
                pose = PoseStamped()
                pose.header.frame_id = from_frame
                if hasattr(stamp, 'to_msg'):
                    pose.header.stamp = stamp.to_msg()
                elif isinstance(stamp, MsgTime):
                    pose.header.stamp = stamp
                else:
                    pose.header.stamp = MsgTime()
                pose.pose.position.x = float(position[0])
                pose.pose.position.y = float(position[1])
                pose.pose.position.z = float(position[2])
                pose.pose.orientation.x = float(orientation[0])
                pose.pose.orientation.y = float(orientation[1])
                pose.pose.orientation.z = float(orientation[2])
                pose.pose.orientation.w = float(orientation[3])
                # Try with message timestamp
                if self.tf_buffer.can_transform(to_frame, from_frame, pose.header.stamp):
                    tf_pose = self.tf_buffer.transform(pose, to_frame, timeout=rclpy.duration.Duration(seconds=0.5))
                    pos = [tf_pose.pose.position.x, tf_pose.pose.position.y, tf_pose.pose.position.z]
                    quat = [tf_pose.pose.orientation.x, tf_pose.pose.orientation.y, tf_pose.pose.orientation.z, tf_pose.pose.orientation.w]
                    return pos, quat, True
                else:
                    print("Trying with latest transform (time=0)")
                    pose.header.stamp = Time().to_msg()
                    if self.tf_buffer.can_transform(to_frame, from_frame, pose.header.stamp):
                        tf_pose = self.tf_buffer.transform(pose, to_frame, timeout=rclpy.duration.Duration(seconds=0.5))
                        pos = [tf_pose.pose.position.x, tf_pose.pose.position.y, tf_pose.pose.position.z]
                        quat = [tf_pose.pose.orientation.x, tf_pose.pose.orientation.y, tf_pose.pose.orientation.z, tf_pose.pose.orientation.w]
                        return pos, quat, True
                    else:
                        print(f"No transform from {from_frame} to {to_frame} at time {stamp} or latest")
                        return position, orientation, False
            except Exception as e:
                print(f"Transform error: {e}")
                return position, orientation, False

        def transform_imu_orientation(orientation, from_frame, to_frame, stamp):
            print(f"Requesting transform from '{from_frame}' to '{to_frame}' at time {stamp}")
            if self.tf_buffer is None or to_frame is None or to_frame == "Select frame...":
                return orientation, True
            if not from_frame:
                print("No frame_id in IMU message!")
                return orientation, False
            try:
                # Try with message timestamp
                if self.tf_buffer.can_transform(to_frame, from_frame, stamp):
                    print("Transform available at message time.")
                    tf = self.tf_buffer.lookup_transform(to_frame, from_frame, stamp, timeout=rclpy.duration.Duration(seconds=0.5))
                else:
                    print("Trying with latest transform (time=0)")
                    stamp = Time().to_msg()
                    if self.tf_buffer.can_transform(to_frame, from_frame, stamp):
                        print("Transform available at latest time.")
                        tf = self.tf_buffer.lookup_transform(to_frame, from_frame, stamp, timeout=rclpy.duration.Duration(seconds=0.5))
                    else:
                        print(f"No transform from {from_frame} to {to_frame} at time {stamp} or latest")
                        return orientation, False
                tf_q = [
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w
                ]
                q_out = quaternion_multiply(tf_q, orientation)
                return q_out, True
            except Exception as e:
                print(f"IMU orientation transform error: {e}")
                return orientation, False

        # --- Subscription callbacks ---

        if msg_type == 'Imu':
            def cb(msg):
                self.last_imu_msg = msg
                self.last_odom_msg = None
                q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
                pos = [0, 0, 0]
                frame_id = getattr(msg.header, "frame_id", "")
                stamp = getattr(msg.header, "stamp", rclpy.time.Time())
                error = False
                # If a frame is selected, transform only the orientation
                if self.selected_frame:
                    q, ok = transform_imu_orientation(q, frame_id, self.selected_frame, stamp)
                    error = not ok
                # If transform failed, show error in GUI
                if error:
                    for edit in self.quat_edits:
                        edit.blockSignals(True)
                        edit.setText("ERROR")
                        edit.blockSignals(False)
                else:
                    for i, val in enumerate(q):
                        self.quat_edits[i].blockSignals(True)
                        self.quat_edits[i].setText(str(val))
                        self.quat_edits[i].blockSignals(False)
                self.on_quat_changed()
                if self.extra_fields_visible:
                    self.update_extra_fields(msg)
            self.imu_sub = self.ros_node.create_subscription(Imu, topic, cb, qos)
        else:
            def cb(msg):
                self.last_odom_msg = msg
                self.last_imu_msg = None
                q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                     msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                frame_id = getattr(msg.header, "frame_id", "")
                stamp = getattr(msg.header, "stamp", rclpy.time.Time())
                error = False
                # If a frame is selected, transform full pose
                if self.selected_frame:
                    pos, q, ok = transform_pose_quat(pos, q, frame_id, self.selected_frame, stamp)
                    error = not ok
                if error:
                    for edit in self.quat_edits:
                        edit.blockSignals(True)
                        edit.setText("ERROR")
                        edit.blockSignals(False)
                else:
                    for i, val in enumerate(q):
                        self.quat_edits[i].blockSignals(True)
                        self.quat_edits[i].setText(str(val))
                        self.quat_edits[i].blockSignals(False)
                self.on_quat_changed()
                if self.extra_fields_visible:
                    self.update_extra_fields(msg)
            self.imu_sub = self.ros_node.create_subscription(Odometry, topic, cb, qos)

        self.load_btn.setText("Stop Stream")

    ##
    # @brief Stop streaming IMU data.
    def stop_stream(self):
        # Remove subscription if it exists
        if self.imu_sub is not None:
            self.ros_node.destroy_subscription(self.imu_sub)
            self.imu_sub = None
        self.load_btn.setText("Start Stream")

    ##
    # @brief Toggle visibility of extra IMU fields.
    def toggle_extra_fields(self):
        self.extra_fields_visible = not self.extra_fields_visible
        self.extra_fields_group.setVisible(self.extra_fields_visible)
        # If showing extra fields, update and resize window
        if self.extra_fields_visible:
            self.expand_btn.setText("Hide Extra Fields")
            self.update_extra_fields()
            self.resize(1000, 800)
        # If hiding extra fields, resize window
        else:
            self.expand_btn.setText("Show Extra Fields")
            self.resize(800, 600)

    ##
    # @brief Update extra IMU fields display.
    # @param msg Optional IMU message.
    def update_extra_fields(self, msg=None):
        # Remove old widgets from layout
        for widget in self.extra_fields_widgets:
            self.extra_fields_layout.removeWidget(widget)
            widget.deleteLater()
        self.extra_fields_widgets = []

        # Determine which message to use (IMU or Odometry)
        m_imu = getattr(self, 'last_imu_msg', None)
        m_odom = getattr(self, 'last_odom_msg', None)

        # If IMU message is available, show IMU fields
        if m_imu is not None:
            columns_layout = QHBoxLayout()
            lin_accel_layout = QVBoxLayout()
            ang_vel_layout = QVBoxLayout()
            lin_fields = [
                ("Linear Acceleration X", f"{m_imu.linear_acceleration.x:.6f}"),
                ("Linear Acceleration Y", f"{m_imu.linear_acceleration.y:.6f}"),
                ("Linear Acceleration Z", f"{m_imu.linear_acceleration.z:.6f}"),
            ]
            ang_fields = [
                ("Angular Velocity X", f"{m_imu.angular_velocity.x:.6f}"),
                ("Angular Velocity Y", f"{m_imu.angular_velocity.y:.6f}"),
                ("Angular Velocity Z", f"{m_imu.angular_velocity.z:.6f}"),
            ]
            # Add linear acceleration fields
            for label, value in lin_fields:
                h = QHBoxLayout()
                lbl = QLabel(label)
                edit = QLineEdit(value)
                edit.setReadOnly(True)
                h.addWidget(lbl)
                h.addWidget(edit)
                container = QWidget()
                container.setLayout(h)
                lin_accel_layout.addWidget(container)
                self.extra_fields_widgets.append(container)
            # Add angular velocity fields
            for label, value in ang_fields:
                h = QHBoxLayout()
                lbl = QLabel(label)
                edit = QLineEdit(value)
                edit.setReadOnly(True)
                h.addWidget(lbl)
                h.addWidget(edit)
                container = QWidget()
                container.setLayout(h)
                ang_vel_layout.addWidget(container)
                self.extra_fields_widgets.append(container)
            columns_layout.addLayout(lin_accel_layout)
            columns_layout.addLayout(ang_vel_layout)
            # Remove old layout widgets if any
            while self.extra_fields_layout.count():
                item = self.extra_fields_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            self.extra_fields_layout.addLayout(columns_layout)
        # If Odometry message is available, show Odometry fields
        elif m_odom is not None:
            odom_fields = [
                ("Position X", f"{m_odom.pose.pose.position.x:.6f}"),
                ("Position Y", f"{m_odom.pose.pose.position.y:.6f}"),
                ("Position Z", f"{m_odom.pose.pose.position.z:.6f}"),
                ("Orientation X", f"{m_odom.pose.pose.orientation.x:.6f}"),
                ("Orientation Y", f"{m_odom.pose.pose.orientation.y:.6f}"),
                ("Orientation Z", f"{m_odom.pose.pose.orientation.z:.6f}"),
                ("Orientation W", f"{m_odom.pose.pose.orientation.w:.6f}"),
                ("Linear Velocity X", f"{m_odom.twist.twist.linear.x:.6f}"),
                ("Linear Velocity Y", f"{m_odom.twist.twist.linear.y:.6f}"),
                ("Linear Velocity Z", f"{m_odom.twist.twist.linear.z:.6f}"),
                ("Angular Velocity X", f"{m_odom.twist.twist.angular.x:.6f}"),
                ("Angular Velocity Y", f"{m_odom.twist.twist.angular.y:.6f}"),
                ("Angular Velocity Z", f"{m_odom.twist.twist.angular.z:.6f}"),
            ]
            layout = QVBoxLayout()
            # Add all odometry fields
            for label, value in odom_fields:
                h = QHBoxLayout()
                lbl = QLabel(label)
                edit = QLineEdit(value)
                edit.setReadOnly(True)
                h.addWidget(lbl)
                h.addWidget(edit)
                container = QWidget()
                container.setLayout(h)
                layout.addWidget(container)
                self.extra_fields_widgets.append(container)
            # Remove old layout widgets if any
            while self.extra_fields_layout.count():
                item = self.extra_fields_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            self.extra_fields_layout.addLayout(layout)
        # If no message, show empty fields
        else:
            layout = QVBoxLayout()
            for label in [
                "Position X", "Position Y", "Position Z",
                "Orientation X", "Orientation Y", "Orientation Z", "Orientation W",
                "Linear Velocity X", "Linear Velocity Y", "Linear Velocity Z",
                "Angular Velocity X", "Angular Velocity Y", "Angular Velocity Z"
            ]:
                h = QHBoxLayout()
                lbl = QLabel(label)
                edit = QLineEdit("")
                edit.setReadOnly(True)
                h.addWidget(lbl)
                h.addWidget(edit)
                container = QWidget()
                container.setLayout(h)
                layout.addWidget(container)
                self.extra_fields_widgets.append(container)
            while self.extra_fields_layout.count():
                item = self.extra_fields_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            self.extra_fields_layout.addLayout(layout)

    ##
    # @brief Refresh the list of available IMU and Odometry topics.
    def refresh_topics(self):
        self.combo.clear()
        self.combo.addItem("Select topic...")
        self.topic_types = {}  # Map topic name to type
        # Loop through all found topics and add to combo box
        topics = self.ros_node.find_imu_topics()
        for t, typ in topics:
            self.combo.addItem(t)
            self.topic_types[t] = typ

    ##
    # @brief Spin the ROS2 node once (process callbacks).
    def spin_ros(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0)

    ##
    # @brief Handle changes in quaternion text boxes.
    def on_quat_changed(self):
        try:
            # Read quaternion values from text boxes
            q = [float(edit.text()) for edit in self.quat_edits]
            # Convert to Euler angles
            e = quaternion_to_euler(q)
            # Update Euler angle text boxes
            for i, val in enumerate(e):
                self.euler_edits[i].setText(f"{val:.2f}")
            self.update_graph()
        except Exception:
            pass

    ##
    # @brief Handle changes in Euler angle text boxes.
    def on_euler_changed(self):
        try:
            # Read Euler angles from text boxes
            e = [float(edit.text()) for edit in self.euler_edits]
            # Convert to quaternion
            q = euler_to_quaternion(e)
            # Update quaternion text boxes
            for i, val in enumerate(q):
                self.quat_edits[i].blockSignals(True)
                self.quat_edits[i].setText(f"{val:.6f}")
                self.quat_edits[i].blockSignals(False)
            self.update_graph()
        except Exception:
            pass

    ##
    # @brief Update the 3D graph visualization.
    def update_graph(self):
        self.graph_widget.clear()
        try:
            # Draw axes
            axis_len = 1.5
            # X axis (red)
            x_axis = gl.GLLinePlotItem(pos=[(0,0,0), (axis_len,0,0)], color=(1,0,0,1), width=2, antialias=True)
            self.graph_widget.addItem(x_axis)
            # Y axis (green)
            y_axis = gl.GLLinePlotItem(pos=[(0,0,0), (0,axis_len,0)], color=(0,1,0,1), width=2, antialias=True)
            self.graph_widget.addItem(y_axis)
            # Z axis (blue)
            z_axis = gl.GLLinePlotItem(pos=[(0,0,0), (0,0,axis_len)], color=(0,0,1,1), width=2, antialias=True)
            self.graph_widget.addItem(z_axis)

            # Add axis labels
            x_label = gl.GLTextItem(pos=(axis_len, 0, 0), text='X', color=(1,0,0,1))
            y_label = gl.GLTextItem(pos=(0, axis_len, 0), text='Y', color=(0,1,0,1))
            z_label = gl.GLTextItem(pos=(0, 0, axis_len), text='Z', color=(0,0,1,1))
            self.graph_widget.addItem(x_label)
            self.graph_widget.addItem(y_label)
            self.graph_widget.addItem(z_label)

            # Draw the rotated vector (magenta for contrast)
            # Read quaternion from text boxes
            q = [float(edit.text()) for edit in self.quat_edits]
            v = [1, 0, 0]
            # Rotate vector by quaternion
            v_rot = rotate_vector_by_quaternion(v, q)
            pts = [ [0, 0, 0], v_rot ]
            pos = [tuple(pts[0]), tuple(pts[1])]
            angle_line = gl.GLLinePlotItem(pos=pos, color=(1,0,1,1), width=4, antialias=True)
            self.graph_widget.addItem(angle_line)
        except Exception:
            pass

    ##
    # @brief Handle widget close event (shutdown ROS2).
    # @param event Close event.
    def closeEvent(self, event):
        rclpy.shutdown()
        event.accept()

##
# @brief Main entry point for the application.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = QuaternionEulerWidget()
    w.show()
    sys.exit(app.exec())
