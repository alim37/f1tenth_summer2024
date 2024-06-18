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
        
       
       
    def scan_callback(self, msg):
        
        # obtain laser scans and preprocess them
        ranges = np.array(msg.ranges)
        self.range_max = msg.range_max
        ranges = np.clip(ranges, 0, self.range_max)
        
        min_angle = -65 / 180.0 * math.pi
        min_idx = int(np.floor((min_angle - msg.angle_min) / msg.angle_increment))
        max_angle = 65 / 180.0 * math.pi
        max_idx = int(np.ceil((max_angle - msg.angle_min) / msg.angle_increment))
        
        # Find the closest point in the LiDAR ranges array
        closest_idx = min_idx
        closest_distance = msg.range_max * 5
        for i in range(min_idx, max_idx+1):
            distance = ranges[i-2] + ranges[i-1] + ranges[i] + ranges[i+1] + ranges[i+2]
            if distance < closest_distance:
                closest_distance = distance
                closest_idx = i

        # eliminate all points inside bubble
        radius = 150
        for i in range(closest_idx - radius, closest_idx + radius):
            ranges[i] = 0.0
        
        # return start index and end index of max gap in free space ranges
        start_gap = min_idx
        end_gap = min_idx
        current_start = min_idx - 1 # if longest gap is at first index this verifies it can be handled
        duration = 0
        longest_duration = 0
        
        for i in range(min_idx, max_idx + 1): # w/out +1 it leaves out last point
            if current_start < min_idx:     # checks if first index has first gap
                if ranges[i] > 0.0:
                    current_start = i
            elif ranges[i] <= 0.0:
                duration = i - current_start
                if duration > longest_duration:
                    longest_duration = duration
                    start_gap = current_start
                    end_gap = i -1
        if current_start >= min_idx:        # checks final gap
            duration = max_idx + 1 - current_start
            if duration > longest_duration:
                longest_duration = duration
                start_gap = current_start
                end_gap = max_idx
                
        # return index of best 'goal' in the gap
        current_max = 0
        for i in range(start_gap, end_gap + 1):
            if ranges[i] > current_max:
                current_max = ranges[i]
                self.angle = msg.angle_min + i * msg.angle_increment
            elif ranges[i] == current_max:
                if abs(msg.angle_min + i * msg.angle_increment) < abs(self.angle):
                    self.angle = msg.angle_min + i * msg.angle_increment
        
        self.reactive_control()
        
    def reactive_control(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = self.angle
        drive_msg.drive.speed = 1.0
        self.drive_pub.publish(drive_msg)
         

def main(args=None):
    rclpy.init(args=args)   
    node = GapFollowNode()     
    rclpy.spin(node)
    rclpy.shutdown()        


if __name__ == "__main__":
    main()
