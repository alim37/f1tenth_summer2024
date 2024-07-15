#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import numpy as np
import math
import csv
import os

class RISEControlNode(Node):      

    def __init__(self):
        super().__init__("rise_controller")
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        

        #self.ks = 0.5
        self.ks = 1.0
        #self.alpha = 1.0
        self.alpha = 0.7
        #self.beta = 0.05
        self.beta = 0.03
        self.desired_distance = 1.0
        self.k = 10.0
        self.prev_time = self.get_clock().now()

        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_error = 0.0

        self.max_steering_angle = 1.0

        self.u = 0
        self.max_integral = 1.0
        #self.max_integral = 0.5

        self.car_theta = 0.0
        self.car_x = 0.0
        self.car_y = 0.0

        self.velocities = []
        self.times = []
        self.start_time = None
        self.lap_end_time = None
        self.lap_completed = False

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

        #self.get_logger().info(f'Error: {error:.2f}, Control: {control_out:.2f}, Distance: {self.desired_distance - error:.2f}')

        self.car_theta += control_out * dt
        self.car_x += 1.5 * math.cos(self.car_theta) * dt
        self.car_y += 1.5 * math.sin(self.car_theta) * dt

    def odom_callback(self, msg):
        if self.start_time is None:
            self.start_time = self.get_clock().now()

        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y

        total_velocity = math.sqrt(v_x**2 + v_y**2)
        self.velocities.append(total_velocity)
        self.times.append(current_time)

        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y

        if current_x <= 1 and current_y <= 1 and not self.lap_completed:
            lap_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.lap_completed = True
            self.get_logger().info(f'Lap completed in {lap_time:.2f} seconds')
        
        if current_x and current_y <= 1:
            self.lap_completed = False

    def save_data(self):
        save_dir = os.path.expanduser('~/data_rise')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save velocities
        velocities_path = os.path.join(save_dir, 'velocities_rise.csv')
        with open(velocities_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Velocity'])  # Header
            for velocity in self.velocities:
                writer.writerow([velocity])
        
        # Save times
        times_path = os.path.join(save_dir, 'times_rise.csv')
        with open(times_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time'])  # Header
            for time in self.times:
                writer.writerow([time])

        self.get_logger().info(f'Velocities saved to: {velocities_path}')
        self.get_logger().info(f'Times saved to: {times_path}')



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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node stopped cleanly.")
    finally:
        node.save_data()
        rclpy.shutdown()        


if __name__ == "__main__":
    main()
