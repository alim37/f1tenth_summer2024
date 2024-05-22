#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped

class TalkerNode(Node):     

    def __init__(self):
        super().__init__("talker")
        
        self.declare_parameter("v", 10.0)
        self.declare_parameter("d", 1000.0)
        self.v_ = self.get_parameter("v").value
        self.d_ = self.get_parameter("d").value
        
        self.publisher_ = self.create_publisher(AckermannDriveStamped, "drive", 10)
        self.timer = self.create_timer(0.1, self.ackermann_publisher)
        
        self.get_logger().info("Talker publisher started.")
        
    def ackermann_publisher(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = self.v_
        msg.drive.steering_angle = self.d_
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: speed={self.v_}, steering_angle={self.d_}')
        

def main(args=None):
    rclpy.init(args=args)   
    node = TalkerNode()     
    rclpy.spin(node)
    rclpy.shutdown()        


if __name__ == "__main__":
    main()