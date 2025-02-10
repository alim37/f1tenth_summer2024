#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from dataclasses import dataclass
from typing import List, Tuple
import math
import os  ### ADDED: Import os for file and directory operations

GRID_RESOLUTION = 0.1       
MAP_WIDTH = 30.0            
MAP_HEIGHT = 30.0           
VEHICLE_WIDTH = 0.3         
WALL_MARGIN = 0.2           

MAX_ITERATIONS = 500
GOAL_TOLERANCE = 0.5
STEER_DISTANCE = 0.8        
LOOKAHEAD_DISTANCE = 1.0    

@dataclass
class RRTNode:
    x: float
    y: float
    parent_idx: int = -1

class RRTPlanner(Node):
    def __init__(self):
        super().__init__('rrt_planner')
        
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        self.grid_width = int(MAP_WIDTH / GRID_RESOLUTION)
        self.grid_height = int(MAP_HEIGHT / GRID_RESOLUTION)
        self.occupancy_grid = np.ones((self.grid_width, self.grid_height), dtype=bool)
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_heading = 0.0
        
        # --- Fixed goal points (waypoints) ---
        # f1
        #self.goals = [(8.79, 0.20), (9.5, 0.71), (9.94, 1.6), (9.5, 7.0), (-13.0, 9.1), (-14.0, 0.0)]
        # default
        #self.goals = [(9.5, 0), (9.5, 7.0), (-13.0, 9.1), (-14.0, 0.0)]
        # outside
        self.goals = [(8.94, -0.43), (10.11, 0.35), (9.5, 7.0), (-13.0, 9.1), (-14.0, 0.0)]
        self.goal_index = 0
        self.goal_x, self.goal_y = self.goals[self.goal_index]
        
        self.prev_steering = 0.0

        self.has_valid_scan = False  ### ADDED: Initialize has_valid_scan
        
        # ### ADDED: Create folder "data_rrt" in the home directory and open a file for velocity logging.
        home_dir = os.path.expanduser("~")
        self.data_dir = os.path.join(home_dir, "data_rrt")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.vel_file = open(os.path.join(self.data_dir, "velocity.txt"), "a")
        # ### END ADDED

        self.get_logger().info('RRT Planner initialized')

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        grid_x = int((x + MAP_WIDTH / 2) / GRID_RESOLUTION)
        grid_y = int((y + MAP_HEIGHT / 2) / GRID_RESOLUTION)
        return np.clip(grid_x, 0, self.grid_width - 1), np.clip(grid_y, 0, self.grid_height - 1)

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices back to world coordinates."""
        x = grid_x * GRID_RESOLUTION - MAP_WIDTH / 2
        y = grid_y * GRID_RESOLUTION - MAP_HEIGHT / 2
        return x, y

    def scan_callback(self, msg: LaserScan):
        """Update the occupancy grid based on incoming laser scan data."""
        self.last_scan_time = self.get_clock().now()
        self.has_valid_scan = True
        
        self.occupancy_grid.fill(True)
        
        for i, distance in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if np.isfinite(distance) and distance < msg.range_max:
                world_x = self.current_x + distance * np.cos(angle + self.current_heading)
                world_y = self.current_y + distance * np.sin(angle + self.current_heading)
                grid_x, grid_y = self.world_to_grid(world_x, world_y)
                
                # # Inflate obstacles by WALL_MARGIN.
                # inflation_radius = int(WALL_MARGIN / GRID_RESOLUTION)
                # x_min = max(0, grid_x - inflation_radius)
                # x_max = min(self.grid_width, grid_x + inflation_radius + 1)
                # y_min = max(0, grid_y - inflation_radius)
                # y_max = min(self.grid_height, grid_y + inflation_radius + 1)
                # self.occupancy_grid[x_min:x_max, y_min:y_max] = False
        
        # Updated safety block:
        # Ensure the vehicle's current position is free (prevent self-collision).
        current_grid_x, current_grid_y = self.world_to_grid(self.current_x, self.current_y)
        safe_radius = int((VEHICLE_WIDTH / 2) / GRID_RESOLUTION)
        x_min = max(0, current_grid_x - safe_radius)
        x_max = min(self.grid_width, current_grid_x + safe_radius + 1)
        y_min = max(0, current_grid_y - safe_radius)
        y_max = min(self.grid_height, current_grid_y + safe_radius + 1)
        self.occupancy_grid[x_min:x_max, y_min:y_max] = True

    def sample_free(self) -> Tuple[float, float]:
        """Sample a random free point from the occupancy grid."""
        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.uniform(-MAP_WIDTH / 2, MAP_WIDTH / 2)
            y = np.random.uniform(-MAP_HEIGHT / 2, MAP_HEIGHT / 2)
            grid_x, grid_y = self.world_to_grid(x, y)
            if self.occupancy_grid[grid_x, grid_y]:
                return x, y
        return self.current_x, self.current_y

    def nearest_node(self, nodes: List[RRTNode], x: float, y: float) -> int:
        """Return the index of the node closest to (x, y)."""
        distances = [(node.x - x)**2 + (node.y - y)**2 for node in nodes]
        return int(np.argmin(distances))

    def steer(self, from_node: RRTNode, to_x: float, to_y: float) -> RRTNode:
        """Generate a new node in the direction from from_node to (to_x, to_y)."""
        dx = to_x - from_node.x
        dy = to_y - from_node.y
        distance = math.hypot(dx, dy)
        if distance > STEER_DISTANCE:
            dx *= STEER_DISTANCE / distance
            dy *= STEER_DISTANCE / distance
        return RRTNode(x=from_node.x + dx, y=from_node.y + dy, parent_idx=-1)

    def is_collision_free(self, from_node: RRTNode, to_node: RRTNode) -> bool:
        """Check if the path between two nodes is free of obstacles."""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        distance = math.hypot(dx, dy)
        num_points = max(50, int(distance / (GRID_RESOLUTION * 2)))
        for t in np.linspace(0, 1, num_points):
            x = from_node.x + t * dx
            y = from_node.y + t * dy
            grid_x, grid_y = self.world_to_grid(x, y)
            if not self.occupancy_grid[grid_x, grid_y]:
                return False
        return True

    def plan_path(self) -> List[RRTNode]:
        """Generate an RRT path from the current position to the global goal."""
        if not self.has_valid_scan:
            return []
        nodes = [RRTNode(x=self.current_x, y=self.current_y)]
        for _ in range(MAX_ITERATIONS):
            # With 20% probability, bias the sample to the global goal.
            if np.random.random() < 0.2:
                sample_x, sample_y = self.goal_x, self.goal_y
            else:
                sample_x, sample_y = self.sample_free()
            nearest_idx = self.nearest_node(nodes, sample_x, sample_y)
            nearest = nodes[nearest_idx]
            new_node = self.steer(nearest, sample_x, sample_y)
            new_node.parent_idx = nearest_idx
            if self.is_collision_free(nearest, new_node):
                nodes.append(new_node)
                
                if math.hypot(new_node.x - self.goal_x, new_node.y - self.goal_y) < GOAL_TOLERANCE:
                    break
        path = []
        if len(nodes) > 1:
            current_idx = len(nodes) - 1
            while current_idx != -1:
                path.append(nodes[current_idx])
                current_idx = nodes[current_idx].parent_idx
            path.reverse()
        return path

    def send_drive_command(self, target_x: float, target_y: float):
        """Publish a drive command based on a pure-pursuit controller.
           The blending of the target with a lookahead (used to slow down turns) has been removed
           so that the vehicle always steers directly toward the target."""
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - self.current_heading

        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        max_steering = 0.4
        new_steering = np.clip(heading_error, -max_steering, max_steering)
        
        # low-pass filter smooths the steering angle, too much oscillation previously
        alpha = 0.7  
        filtered_steering = alpha * new_steering + (1 - alpha) * self.prev_steering
        self.prev_steering = filtered_steering

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = filtered_steering

        speed = 1.5
        drive_msg.drive.speed = speed

        self.drive_pub.publish(drive_msg)

    def odom_callback(self, msg: Odometry):
        """Update vehicle state, check waypoint progress, plan/control a path, and record velocity."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        self.current_heading = math.atan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                                          1.0 - 2.0 * (quat.y**2 + quat.z**2))
        
        # check if goal has been reached, hypot = euclidean norm
        if math.hypot(self.current_x - self.goal_x, self.current_y - self.goal_y) < GOAL_TOLERANCE:
            self.goal_index = (self.goal_index + 1) % len(self.goals)
            self.goal_x, self.goal_y = self.goals[self.goal_index]

        # ### ADDED: Record the vehicle's velocity to file.
        # Extract linear velocity from odometry (assuming forward velocity is in twist.twist.linear.x)
        velocity = msg.twist.twist.linear.x
        timestamp = self.get_clock().now().nanoseconds / 1e9  # convert to seconds as a float
        self.vel_file.write(f"{timestamp},{velocity}\n")
        self.vel_file.flush()  # Ensure the data is written immediately
        # ### END ADDED

        path = self.plan_path()
        if path:
            # choosing best node with minimal heading error
            best_node = None
            best_error = float('inf')
            for node in path:
                dx = node.x - self.current_x
                dy = node.y - self.current_y
                dist = math.hypot(dx, dy)
                if dist < LOOKAHEAD_DISTANCE:
                    continue
                # Compute the angle from the car to the node.
                node_angle = math.atan2(dy, dx)
                error = abs(math.atan2(math.sin(node_angle - self.current_heading),
                                       math.cos(node_angle - self.current_heading)))

                if error < best_error:
                    best_error = error
                    best_node = node
            if best_node is None:
                best_node = path[-1]
            self.send_drive_command(best_node.x, best_node.y)

    def destroy_node(self):
        # ### ADDED: Close the velocity file when shutting down.
        self.vel_file.close()
        # ### END ADDED
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RRTPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
