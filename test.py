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
import math

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy

def quaternion_to_euler(q):
    # q = [x, y, z, w]
    x, y, z, w = q

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    # Convert to degrees
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

def euler_to_quaternion(e):
    # e = [roll, pitch, yaw] in degrees
    roll, pitch, yaw = [math.radians(a) for a in e]
    
    x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return x, y, z, w

def rotate_vector_by_quaternion(v, q):
    # v: [x, y, z], q: [x, y, z, w]
    # Quaternion multiplication: q * v * q_conj
    x, y, z = v
    qx, qy, qz, qw = q
    # Quaternion representing the vector
    vx, vy, vz, vw = x, y, z, 0
    # q * v
    rx = qw * vx + qy * vz - qz * vy
    ry = qw * vy + qz * vx - qx * vz
    rz = qw * vz + qx * vy - qy * vx
    rw = -qx * vx - qy * vy - qz * vz
    # result * q_conj
    cx = -qx
    cy = -qy
    cz = -qz
    cw = qw
    fx = rx * cw + rw * cx + ry * cz - rz * cy
    fy = ry * cw + rw * cy + rz * cx - rx * cz
    fz = rz * cw + rw * cz + rx * cy - ry * cx
    return [fx, fy, fz]

class ROS2IMUFinder(Node):
    def __init__(self):
        super().__init__('imu_finder')
        self.topics = []

    def find_imu_topics(self):
        self.topics = []
        topic_list = self.get_topic_names_and_types()
        for name, types in topic_list:
            if 'sensor_msgs/msg/Imu' in types:
                self.topics.append(name)
        return self.topics

class RefreshingComboBox(QComboBox):
    def __init__(self, refresh_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refresh_callback = refresh_callback

    def showPopup(self):
        self.refresh_callback()
        super().showPopup()

class QuaternionEulerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quaternion/Euler Angle Visualizer")
        self.resize(800, 600)
        self.imu_sub = None
        self.extra_fields_visible = False
        self.extra_fields_widgets = []
        self.init_ui()
        self.init_ros()
        self.update_graph()

        # Timer to spin ROS2 node
        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.spin_ros)
        self.ros_timer.start(50)  # 20 Hz

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Angle input layout
        angle_layout = QHBoxLayout()

        # Euler angles
        euler_group = QGroupBox("Euler Angles (deg)")
        euler_layout = QVBoxLayout()
        self.euler_labels = []
        self.euler_edits = []
        for name in ['Roll', 'Pitch', 'Yaw']:
            h = QHBoxLayout()
            lbl = QLabel(name)
            edit = QLineEdit("0.0")
            edit.setReadOnly(False)  # Make Euler text boxes editable at startup
            edit.editingFinished.connect(self.on_euler_changed)  # Update quaternion when Euler is changed
            h.addWidget(lbl)
            h.addWidget(edit)
            euler_layout.addLayout(h)
            self.euler_labels.append(lbl)
            self.euler_edits.append(edit)
        euler_group.setLayout(euler_layout)
        angle_layout.addWidget(euler_group)

        # Quaternion angles
        quat_group = QGroupBox("Quaternion (x, y, z, w)")
        quat_layout = QVBoxLayout()
        self.quat_labels = []
        self.quat_edits = []
        for name in ['x', 'y', 'z', 'w']:
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

        # 3D Graph only (remove Artificial Horizon)
        graph_layout = QHBoxLayout()

        # 3D Graph
        self.graph_widget = gl.GLViewWidget()
        self.graph_widget.setCameraPosition(distance=5)
        graph_layout.addWidget(self.graph_widget, stretch=1)

        main_layout.addLayout(graph_layout)

        # Combo box and button
        bottom_layout = QHBoxLayout()
        self.combo = RefreshingComboBox(self.refresh_topics)
        self.combo.setEditable(False)
        self.combo.addItem("Select IMU topic...")
        self.load_btn = QPushButton("Start Stream")
        self.expand_btn = QPushButton("Show Extra IMU Fields")
        bottom_layout.addWidget(self.combo)
        bottom_layout.addWidget(self.load_btn)
        bottom_layout.addWidget(self.expand_btn)
        main_layout.addLayout(bottom_layout)

        self.load_btn.clicked.connect(self.toggle_stream)
        self.expand_btn.clicked.connect(self.toggle_extra_fields)

        # Extra fields area (hidden by default)
        self.extra_fields_group = QGroupBox("Other IMU Values")
        self.extra_fields_layout = QVBoxLayout()
        self.extra_fields_group.setLayout(self.extra_fields_layout)
        self.extra_fields_group.setVisible(False)
        main_layout.addWidget(self.extra_fields_group)

        self.setLayout(main_layout)

    def toggle_stream(self):
        if self.load_btn.text() == "Start Stream":
            self.set_textboxes_editable(False)  # Make uneditable when streaming
            self.start_stream()
        else:
            self.set_textboxes_editable(True)   # Make editable when not streaming
            self.stop_stream()

    def set_textboxes_editable(self, editable):
        # Quaternion text boxes
        for edit in self.quat_edits:
            edit.setReadOnly(not editable)
        # Euler text boxes (usually read-only, but included for completeness)
        for edit in self.euler_edits:
            edit.setReadOnly(not editable)  # Euler angles should always be read-only
        # Extra fields (always read-only)
        if hasattr(self, 'extra_fields_widgets'):
            for container in self.extra_fields_widgets:
                for child in container.findChildren(QLineEdit):
                    child.setReadOnly(True)

    def start_stream(self):
        topic = self.combo.currentText()
        if topic == "Select IMU topic...":
            return

        # Remove previous subscription if any
        if self.imu_sub is not None:
            self.ros_node.destroy_subscription(self.imu_sub)
            self.imu_sub = None

        # Use BEST_EFFORT reliability
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        def cb(msg):
            self.last_imu_msg = msg  # Save the latest message
            q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            for i, val in enumerate(q):
                self.quat_edits[i].blockSignals(True)
                self.quat_edits[i].setText(str(val))
                self.quat_edits[i].blockSignals(False)
            self.on_quat_changed()
            if self.extra_fields_visible:
                self.update_extra_fields(msg)
        self.imu_sub = self.ros_node.create_subscription(Imu, topic, cb, qos)
        self.load_btn.setText("Stop Stream")

    def stop_stream(self):
        if self.imu_sub is not None:
            self.ros_node.destroy_subscription(self.imu_sub)
            self.imu_sub = None
        self.load_btn.setText("Start Stream")

    def toggle_extra_fields(self):
        self.extra_fields_visible = not self.extra_fields_visible
        self.extra_fields_group.setVisible(self.extra_fields_visible)
        if self.extra_fields_visible:
            self.expand_btn.setText("Hide Extra IMU Fields")
            self.update_extra_fields()
            self.resize(1000, 800)
        else:
            self.expand_btn.setText("Show Extra IMU Fields")
            self.resize(800, 600)

    def update_extra_fields(self, msg=None):
        # Remove old widgets
        for widget in self.extra_fields_widgets:
            self.extra_fields_layout.removeWidget(widget)
            widget.deleteLater()
        self.extra_fields_widgets = []

        # Create horizontal layout for two columns
        columns_layout = QHBoxLayout()
        lin_accel_layout = QVBoxLayout()
        ang_vel_layout = QVBoxLayout()

        # If no message, just show empty fields
        if not hasattr(self, 'last_imu_msg') or self.last_imu_msg is None:
            lin_fields = [
                ("Linear Acceleration X", ""),
                ("Linear Acceleration Y", ""),
                ("Linear Acceleration Z", ""),
            ]
            ang_fields = [
                ("Angular Velocity X", ""),
                ("Angular Velocity Y", ""),
                ("Angular Velocity Z", ""),
            ]
        else:
            m = self.last_imu_msg
            lin_fields = [
                ("Linear Acceleration X", f"{m.linear_acceleration.x:.6f}"),
                ("Linear Acceleration Y", f"{m.linear_acceleration.y:.6f}"),
                ("Linear Acceleration Z", f"{m.linear_acceleration.z:.6f}"),
            ]
            ang_fields = [
                ("Angular Velocity X", f"{m.angular_velocity.x:.6f}"),
                ("Angular Velocity Y", f"{m.angular_velocity.y:.6f}"),
                ("Angular Velocity Z", f"{m.angular_velocity.z:.6f}"),
            ]

        # Add linear acceleration fields (left)
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

        # Add angular velocity fields (right)
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

        # Add both columns to the horizontal layout
        columns_layout.addLayout(lin_accel_layout)
        columns_layout.addLayout(ang_vel_layout)

        # Set the new layout for the extra fields group
        # Remove old layout widgets if any
        while self.extra_fields_layout.count():
            item = self.extra_fields_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.extra_fields_layout.addLayout(columns_layout)

    def init_ros(self):
        rclpy.init(args=None)
        self.ros_node = ROS2IMUFinder()

    def refresh_topics(self):
        self.combo.clear()
        self.combo.addItem("Select IMU topic...")
        topics = self.ros_node.find_imu_topics()
        for t in topics:
            self.combo.addItem(t)

    def spin_ros(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0)

    def on_quat_changed(self):
        try:
            q = [float(edit.text()) for edit in self.quat_edits]
            e = quaternion_to_euler(q)
            for i, val in enumerate(e):
                self.euler_edits[i].setText(f"{val:.2f}")
            self.update_graph()
        except Exception:
            pass

    def on_euler_changed(self):
        try:
            e = [float(edit.text()) for edit in self.euler_edits]
            q = euler_to_quaternion(e)
            for i, val in enumerate(q):
                self.quat_edits[i].blockSignals(True)
                self.quat_edits[i].setText(f"{val:.6f}")
                self.quat_edits[i].blockSignals(False)
            self.update_graph()
        except Exception:
            pass

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
            q = [float(edit.text()) for edit in self.quat_edits]
            v = [1, 0, 0]
            v_rot = rotate_vector_by_quaternion(v, q)
            pts = [ [0, 0, 0], v_rot ]
            pos = [tuple(pts[0]), tuple(pts[1])]
            angle_line = gl.GLLinePlotItem(pos=pos, color=(1,0,1,1), width=4, antialias=True)
            self.graph_widget.addItem(angle_line)
        except Exception:
            pass

    def closeEvent(self, event):
        rclpy.shutdown()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = QuaternionEulerWidget()
    w.show()
    sys.exit(app.exec())