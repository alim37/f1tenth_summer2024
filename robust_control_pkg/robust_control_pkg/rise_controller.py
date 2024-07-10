#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import math

class RISEControlNode(Node):      

    def __init__(self):
        super().__init__("rise_controller")
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.ks = 0.5
        self.alpha = 1.0
        self.beta = 0.05
        self.desired_distance = 1.0
        self.k = 10.0
        self.prev_time = self.get_clock().now()

        self.integral = 0.0
        self.prev_error = 0.0

        self.max_steering_angle = 1.0

        self.u = 0
        self.max_integral = 1.0


    def scan_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = current_time

        a = self.get_range(msg, 45)
        b = self.get_range(msg, 90)
        alpha = math.atan2(a*math.cos(math.radians(45)) - b, a*math.sin(math.radians(45)))
        D_t = b*math.cos(alpha)
        D_t1 = D_t + 1.10*math.sin(alpha)
        error = self.desired_distance - D_t1

        error_dt = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error

        r = error_dt + self.alpha * error

        self.integral += self.beta * np.tanh(self.k * error) * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)

        u_dot = -(self.ks + 1)*r - self.integral

        self.u += u_dot * dt

        control_out = np.clip(self.u, -self.max_steering_angle, self.max_steering_angle)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 1.5
        drive_msg.drive.steering_angle = control_out
        self.drive_pub.publish(drive_msg)

        self.get_logger().info(f'Error: {error:.2f}, Control: {control_out:.2f}, Distance: {self.desired_distance - error:.2f}')

    def get_range(self, data, angle):
        angle = math.radians(angle)

        index = int((angle - data.angle_min) / data.angle_increment)

        if 0 <= index < len(data.ranges):
            return data.ranges[index]
        else:
            return float('inf')
        

def main(args=None):
    rclpy.init(args=args)   
    node = RISEControlNode()      
    rclpy.spin(node)
    rclpy.shutdown()        


if __name__ == "__main__":
    main()
