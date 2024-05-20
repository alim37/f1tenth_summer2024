#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from my_robot_interfaces.msg import HardwareStatus

class HardwaretatusPublisherNode(Node):      # MODIFY NAME

    def __init__(self):
        super().__init__("hardware_status_publisher")  # MODIFY NAME
        self.hw_status_publisher_ = self.create_publisher(HardwareStatus, "hardware_status", 10)
        self.timer_ = self.create_timer(1.0, self.publish_hardware_status)
        self.get_logger().info("Hardware status publisher has been started.")
        
    def publish_hardware_status(self):
        msg = HardwareStatus()
        msg.temperature = 45
        msg.are_motors_ready = True
        msg.debug_message = "Nothing special here"
        self.hw_status_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)   # must be the first line written in any ROS2 program

    node = HardwaretatusPublisherNode()      # MODIFY NAME
    rclpy.spin(node)

    rclpy.shutdown()        # must be last line


if __name__ == "__main__":
    main()