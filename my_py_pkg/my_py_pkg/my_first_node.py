#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

def main(args=None):
    rclpy.init(args=args)   # must be the first line written in any ROS2 program

    node = Node("py_test")
    node.get_logger().info("Hello ROS2")
    rclpy.spin(node)        # continues running program after shutdown (important for future uses)

    rclpy.shutdown()        # must be last line


if __name__ == "__main__":
    main()