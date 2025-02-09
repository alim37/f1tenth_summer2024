#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import math

class DriveStraightNode(Node):
    def __init__(self):
        super().__init__('drive_straight')
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.start_x = None
        self.start_y = None
        
        self.distance_goal = 5.0
        
        self.drive_speed = 1.0 
        self.steering_angle = 0.0  
        
        self.goal_reached = False
        
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        if self.goal_reached:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)
            self.get_logger().info('5-meter goal reached. Stopping the vehicle.')
            rclpy.shutdown()
            return
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.drive_speed
        drive_msg.drive.steering_angle = self.steering_angle
        self.drive_pub.publish(drive_msg)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        if self.start_x is None or self.start_y is None:
            self.start_x = x
            self.start_y = y
            self.get_logger().info(f'Start position set to x: {x:.2f}, y: {y:.2f}')
            return
        
        dx = x - self.start_x
        dy = y - self.start_y
        distance = math.sqrt(dx**2 + dy**2)
        self.get_logger().debug(f'Distance traveled: {distance:.2f} meters')
        
        if distance >= self.distance_goal:
            self.goal_reached = True

def main(args=None):
    rclpy.init(args=args)
    node = DriveStraightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
