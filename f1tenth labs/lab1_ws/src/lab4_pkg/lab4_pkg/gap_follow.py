#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
import math

class GapFollowNode(Node):      

    def __init__(self):
        super().__init__("gap_follow")
        
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
       
    # uses the Disparity Extender algorithm described in F1Tenth by this team: 
    # https://www.nathanotterness.com/2019/04/the-disparity-extender-algorithm-and.html
    
    def scan_callback(self, msg):
        ranges = msg.ranges
        
        disparities = []
        for i in range(1, len(ranges)):
            if abs(ranges[i] - ranges[i-1]) > .2:
                disparities.append(i)
        
        filtered_ranges = list(ranges)
        for disp in disparities:
            if ranges[disp] < ranges[disp-1]:
                shorter = ranges[disp]
                further = ranges[disp-1]
            else:
                shorter = ranges[disp-1]
                further = ranges[disp]
                
            
            safety_bubble = shorter
            for i in range(disp+1, len(ranges)):
                if ranges[i] >= further:
                    break
                filtered_ranges[i] = safety_bubble
                safety_bubble += 0.1
                
        farthest_point = max(filtered_ranges[int(len(ranges)/4):int(3*len(ranges)/4)])
        farthest_index = filtered_ranges.index(farthest_point)
        steering_angle = (farthest_index - len(ranges)/2) * msg.angle_increment
        
        msg = AckermannDriveStamped()
        msg.drive.speed = 1.0
        msg.drive.steering_angle = steering_angle
        
        self.drive_pub.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)   
    node = GapFollowNode()     
    rclpy.spin(node)
    rclpy.shutdown()        


if __name__ == "__main__":
    main()