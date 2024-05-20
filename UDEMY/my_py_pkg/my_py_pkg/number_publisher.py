#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from example_interfaces.msg import Int64

class NumberPublisherNode(Node):      # MODIFY NAME

    def __init__(self):
        super().__init__("number_publisher")  

        # declare parameter to change at runtime, default parameter if you don't change is 2. 
        self.declare_paramter("number_to_publish", 2)
        self.number_ = self.get_parameter("number_to_publish").value
        
        self.publisher_ = self.create_publisher(Int64, "number", 10)
        self.number_timer_ = self.create_timer(1.0, self.publish_number)
        self.get_logger().info("Number publisher has been started.")
    
    def publish_number(self):
        msg = Int64()
        msg.data = self.number_
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)   # must be the first line written in any ROS2 program

    node = NumberPublisherNode()      # MODIFY NAME
    rclpy.spin(node)

    rclpy.shutdown()        # must be last line


if __name__ == "__main__":
    main()
