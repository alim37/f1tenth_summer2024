#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class RelayNode(Node):      

    def __init__(self):
        super().__init__("relay")  
        
        self.talker_subscriber = self.create_subscription(AckermannDriveStamped, "drive", self.drive_callback, 10)
        
        self.publisher_ = self.create_publisher(AckermannDriveStamped, "drive_relay", 10)
        
    def drive_callback(self, msg):
        new_speed = msg.drive.speed * 3
        new_angle = msg.drive.steering_angle * 3
        
        new_drive_msg = AckermannDriveStamped()
        new_drive_msg.header = msg.header
        new_drive_msg.drive.speed = new_speed
        new_drive_msg.drive.steering_angle = new_angle
        
        self.publisher_.publish(new_drive_msg)
        self.get_logger().info(f'Published: speed={new_speed}, steering_angle={new_angle}')

        

def main(args=None):
    rclpy.init(args=args)   
    node = RelayNode()      
    rclpy.spin(node)
    rclpy.shutdown()        


if __name__ == "__main__":
    main()