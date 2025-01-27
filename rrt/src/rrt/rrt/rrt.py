#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
import numpy as np
import math

class RRTNode(Node):

    def __init__(self):
        super().__init__("RRT")

        self.local_map_size = 5.0  
        self.resolution = 0.1  
        self.grid_size = int(self.local_map_size / self.resolution)  

        self.static_layer = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.dynamic_layer = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.laserscan_callback, 10)
        self.occupancy_pub = self.create_publisher(OccupancyGrid, "/local_occupancy_grid", 10)

        self.timer = self.create_timer(0.1, self.publish_occupancy_grid)

        self.create_static_walls()

    def create_static_walls(self):
        """
        Populate the static layer with walls 1 meter on the left and right of the car.
        """
        wall_margin = int(0.1 / self.resolution)  
        left_wall_x = int((-1.0 + self.local_map_size / 2) / self.resolution)
        right_wall_x = int((1.0 + self.local_map_size / 2) / self.resolution)

        for y in range(self.grid_size):
            # Mark left wall
            for x in range(left_wall_x - wall_margin, left_wall_x + wall_margin + 1):
                if 0 <= x < self.grid_size:
                    self.static_layer[y, x] = 1
            # Mark right wall
            for x in range(right_wall_x - wall_margin, right_wall_x + wall_margin + 1):
                if 0 <= x < self.grid_size:
                    self.static_layer[y, x] = 1

    def laserscan_callback(self, msg):
        """
        Populate the dynamic layer based on LaserScan data.
        """
        self.dynamic_layer.fill(0)  

        for i, distance in enumerate(msg.ranges):
            if distance == float('inf') or distance == 0.0:
                continue

            angle = msg.angle_min + i * msg.angle_increment

            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            grid_x = int((x + self.local_map_size / 2) / self.resolution)
            grid_y = int((y + self.local_map_size / 2) / self.resolution)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.dynamic_layer[grid_y, grid_x] = 1

    def publish_occupancy_grid(self):
        """
        Publish the combined static and dynamic layers as an occupancy grid.
        """
        grid_msg = OccupancyGrid()

        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"

        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.grid_size
        grid_msg.info.height = self.grid_size
        grid_msg.info.origin = Pose(
            position=Point(
                x=-self.local_map_size / 2, 
                y=-self.local_map_size / 2,
                z=0.0
            ),
            orientation=Quaternion(
                x=0.0,
                y=0.0,
                z=0.0,
                w=1.0
            )
        )

        combined_layer = np.maximum(self.static_layer, self.dynamic_layer)

        flattened_grid = combined_layer.flatten()
        grid_msg.data = (flattened_grid * 100).tolist()  # Scale binary data to since nav_msgs is range[0,100]

        self.occupancy_pub.publish(grid_msg)
        self.get_logger().info("Published occupancy grid with static and dynamic layers.")

def main(args=None):
    rclpy.init(args=args)
    node = RRTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
