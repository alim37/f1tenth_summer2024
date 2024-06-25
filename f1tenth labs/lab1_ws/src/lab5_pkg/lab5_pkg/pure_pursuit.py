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
            [-60.224998, -51.224998]    # loop to first    
        ]) 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html examples
        self.tck, _ = splprep([targets[:,0], targets[:,1]], s=0)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.lookahead = 1.0
       # self.current_position = np.array([-51.224998, -51.224998]) # initialized to starting
        self.current_position = np.array([0, 0])
        self.theta = 0
        self.wheelbase = 1.0
        
    def scan_callback(self, msg):    
        u = np.linspace(0,1, num = 100)
        spline_points = np.array(splev(u, self.tck)).T
        lookahead_point = self.find_lookahead(spline_points, self.current_position, self.lookahead)
        steering_angle, speed = self.control(lookahead_point, self.current_position, self.theta)
        
        self.update_position(steering_angle, speed)
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)
        
        self.get_logger().info(f"Position: {self.current_position}, Theta: {self.theta}, Steering: {steering_angle}")
        
    def calculate_distance(self, point1, point2):
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return np.sqrt(dx**2 + dy**2)
    
    def find_lookahead(self, points, position, lookahead):
        for point in points:
            distance = self.calculate_distance(point, position)
            if distance >= lookahead:
                return point
        return points[-1]
    
    def control(self, lookahead_point, current_position, theta):
        dx = lookahead_point[0] - current_position[0]
        dy = lookahead_point[1] - current_position[1]
        d = np.sqrt(dx**2 + dy**2)
        alpha = np.arctan2(dy,dx)
        
        heading_error = alpha - theta
        
        steering_angle = np.arctan2(2.0 * self.wheelbase * np.sin(heading_error), d)
        speed = 1.0
        return steering_angle, speed
    
    def update_position(self, steering_angle, speed):
        dt = 0.1  
        self.theta += steering_angle * speed * dt
        self.current_position[0] += speed * np.cos(self.theta) * dt
        self.current_position[1] += speed * np.sin(self.theta) * dt
        

def main(args=None):
    rclpy.init(args=args)  
    node = PurePursuitNode()      
    rclpy.spin(node)
    rclpy.shutdown()        


if __name__ == "__main__":
    main()