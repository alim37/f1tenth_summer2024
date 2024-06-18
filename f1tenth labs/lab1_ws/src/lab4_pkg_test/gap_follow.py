#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class GapFollowNode(Node):      

    def __init__(self):
        super().__init__("gap_follow")
        
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.dispaity_threshold = 0.2
        self.car_width = 0.5
        self.tolerance = 0.2
        
       
       
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        filtered_ranges = self.disparity_extender(ranges)
        
        farthest_distance = 0.0
        target_angle = 0.0
        
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        
        for i, angle in enumerate(angles):
            if -math.pi / 2 <= angle <= math.pi / 2:
                distance = filtered_ranges[i]
                if distance > farthest_distance:
                    farthest_distance = distance
                    target_angle = angle
                    
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 1.0
        drive_msg.drive.steering_angle = target_angle
        self.drive_pub.publish(drive_msg)
        
    
    def disparity_extender(self, ranges):
        filtered_ranges = ranges.copy()
        for i in range(len(ranges) - 1):
            if abs(ranges[i] - ranges[i+1]) > self.dispaity_threshold:
                closer_range = min(ranges[i], ranges[i+1])
                
                samples_to_cover = int((self.car_width / 2 + self.tolerance) / closer_range * len(ranges) / (2 * math.pi))
                
                if ranges[i] > ranges[i+1]:
                    start_idx = i + 1
                else:
                    start_idx = i
                
                for j in range(start_idx, min(start_idx + samples_to_cover, len(ranges))):
                    if ranges[j] > closer_range:
                        filtered_ranges[j] = closer_range
        return filtered_ranges
        
        
         

def main(args=None):
    rclpy.init(args=args)   
    node = GapFollowNode()     
    rclpy.spin(node)
    rclpy.shutdown()        


if __name__ == "__main__":
    main()
