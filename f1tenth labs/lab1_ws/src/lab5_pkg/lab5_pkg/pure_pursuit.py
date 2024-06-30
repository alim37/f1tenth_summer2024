#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from scipy.interpolate import splprep, splev
import numpy as np
import math

class PurePursuitNode(Node):      

    def __init__(self):
        super().__init__("pure_pursuit")  
        
        self.targets = np.array([
            [1, 0],   
            [2, 0],   
            [3, 0],   
            [4, 0],   
            [5, 0],   
            [6, 0],
            [7, 0],
            [8,0],
            #[8.9473, 0],
            [9.5, 0],
            #[9.2211, 0.0591],
            #[9.6555, 0.3607],
            #[9.9239, 1.0871],
           # [9.7781, 2.0921],
            [9.8000, 3.0000],
            [9.8000, 4.0000],
            [9.8000, 5.0000],
            [9.8000, 6.0000],
            [9.8000, 7.0000],
            [9.8000, 7.3779],
            [9.8132, 7.7535],
            [9.5962, 8.1122],
            [8.9921, 8.3980],
            [7.7689, 8.4908],
            [7.0000, 8.5759],
            [6.0000, 8.5759],
            [5.0000, 8.5759],
            [4.0000, 8.5759],
            [3.0000, 8.5759],
            [2.0000, 8.5759],
            [1.0000, 8.5759],
            [0.0000, 8.5759],
            [-1.0000, 8.5759],
            [-2.0000, 8.5759],
            [-3.0000, 8.5759],
            [-4.0000, 8.5759],
            [-5.0000, 8.5759],
            [-6.0000, 8.5759],
            [-7.0000, 8.5759],
            [-8.0000, 8.5759],
            [-9.0000, 8.5759],
            [-10.0000, 8.5759],
            [-11.0000, 8.5759],
            [-12.0000, 8.5759],
            [-12.4353, 8.5759],
            [-12.9307, 8.3914],
            [-13.3309, 7.8879],
            [-13.6016, 6.5028],
            [-13.7000, 6.0000],
            [-13.7000, 5.0000],
            [-13.7000, 4.0000],
            [-13.7000, 3.0000],
            [-13.7000, 2.0000],
            [-13.7000, 1.0000],
            [-13.5205, 0.3887],
            [-13.0705, -0.0388],
            [-12.0000, 0],  
            [-11.0000, 0],
            [-10.0000, 0],
            [-9, 0],
            [-8, 0],  
            [-7, 0], 
            [-6, 0], 
            [-5, 0], 
            [-4, 0], 
            [-3, 0], 
            [-2, 0], 
            [-1, 0],                    
        ]) 
                
        self.current_target_idx = 0

        self.lookahead_distance = 1.0
        self.wheelbase = 0.3
        self.max_steering_angle = 0.5

        self.drive_pub =self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        self.create_timer(0.1, self.pure_pursuit)
        self.current_pose = None
    
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        #self.get_logger().info('test')
        self.get_logger().info(f'Received odom: x={self.current_pose.position.x:.2f}, y={self.current_pose.position.y:.2f}')

    def pure_pursuit(self):
        if self.current_pose is None:
            return
        
        if self.current_target_idx >= len(self.targets):
            self.get_logger().info('Reached final target')
            return
        
        current_target = self.targets[self.current_target_idx]

        dx = current_target[0] - self.current_pose.position.x
        dy = current_target[1] - self.current_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < self.lookahead_distance:
            self.current_target_idx += 1
            return
        
        y = dy 
        steering_angle = 2 * abs(y) / (self.lookahead_distance**2)
        steering_angle = min(steering_angle, self.max_steering_angle)
        
        if y < 0:
            steering_angle = -steering_angle

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 1.0
        drive_msg.drive.steering_angle = steering_angle

        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)   

    node = PurePursuitNode()      
    rclpy.spin(node)
    rclpy.shutdown()       

if __name__ == "__main__":
    main()
