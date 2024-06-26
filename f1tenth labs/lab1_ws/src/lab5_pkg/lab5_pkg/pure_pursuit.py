#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
import numpy as np
from scipy.interpolate import splprep, splev

class PurePursuitNode(Node):     

    def __init__(self):
        super().__init__("pure_pursuit")
        targets = np.array([
            [-60.224998, -51.224998],   # first corner
            [-60.524998, -53.224998],   # first corner
            [-60.524998, -59.224998],   # second corner
            [-59.524998, -59.524998],   # second corner
            [-37.524998, -59.524998],   # third corner
            [-37.224998, -58.524998],   # third corner
            [-37.224998, -51.524998],   # fourth corner
            [-38.224998, -51.224998],   # fourth corner
            [-60.224998, -51.224998]    # fourth to first    
        ]) 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html examples
        self.tck, _ = splprep([targets[:,0], targets[:,1]], s=0)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.L = 0.3    # wheelbase
        self.kv = 0.5   # lookahead distance gain
        self.current_heading = 0
        self.speed = 1.0
        self.current_position = np.array([-51.224998, -51.224998]) # initialized to starting point
        self.theta = 0
        
    def scan_callback(self, msg):    
        u = np.linspace(0,1, num=100)
        spline_points = np.array(splev(u, self.tck)).T
        
        lookahead_point = self.find_lookahead_point(spline_points)
        
        steering_angle = self.calculate_steering_angle(lookahead_point)
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = steering_angle
        
        self.drive_pub.publish(drive_msg)
        
    def find_lookahead_point(self, points):
        dl = self.kv * self.speed   # eq in book chapter
        
        for point in points:
            distance = np.linalg.norm(point - self.current_position)  # sqrt(x^2 + y^2)
            if distance >= dl:
                return point
        return points[-1]
    
    def calculate_steering_angle(self, lookahead_point):
        dx = lookahead_point[0] - self.current_position[0]
        dy = lookahead_point[1] - self.current_position[1]
        
        lookahead_angle = np.arctan2(dy,dx)
        alpha = lookahead_angle - self.current_heading
        
        dl = np.linalg.norm([dx,dy])
        steering_angle = np.arctan2(2 * self.L * np.sin(alpha), dl)
        
        return steering_angle
         

def main(args=None):
    rclpy.init(args=args)  
    node = PurePursuitNode()      
    rclpy.spin(node)
    rclpy.shutdown()        


if __name__ == "__main__":
    main()
