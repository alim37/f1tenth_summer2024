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

# Configuration parameters
GRID_RESOLUTION = 0.1       # meters per cell
MAP_WIDTH = 30.0            # increased map width in meters
MAP_HEIGHT = 30.0           # increased map height in meters
VEHICLE_WIDTH = 0.3         # meters
WALL_MARGIN = 0.5           # meters buffer from walls

# RRT parameters
MAX_ITERATIONS = 500
GOAL_TOLERANCE = 0.5
STEER_DISTANCE = 0.8        # For finer control
LOOKAHEAD_DISTANCE = 1.0    # For better responsiveness

@dataclass
class RRTNode:
    x: float
    y: float
    parent_idx: int = -1

class RRTPlanner(Node):
    def __init__(self):
        super().__init__('rrt_planner')
        
        # Subscribers and publisher
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10)
        
        # Occupancy grid initialization using global MAP_WIDTH and MAP_HEIGHT
        self.grid_width = int(MAP_WIDTH / GRID_RESOLUTION)
        self.grid_height = int(MAP_HEIGHT / GRID_RESOLUTION)
        self.occupancy_grid = np.ones((self.grid_width, self.grid_height), dtype=bool)
        
        # Current vehicle state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_heading = 0.0
        
        # Global goal (updated dynamically based on current position)
        self.goal_x = 0.0
        self.goal_y = 0.0
        
        # Optional limits (from your goal logic snippet)
        self.x_limit_top = 0.0
        self.x_limit_bot = 0.0
        self.y_limit_left = 0.0
        self.y_limit_right = 0.0
        
        # Safety flags for laser scan
        self.last_scan_time = self.get_clock().now()
        self.scan_timeout = 1.0  # Increased timeout (seconds)
        self.has_valid_scan = False
        
        # Low-pass filter state for steering command
        self.prev_steering = 0.0
        
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
        
        # Assume free space initially.
        self.occupancy_grid.fill(True)
        
        # Process each laser scan measurement.
        # angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment)

        #for angle, distance in zip(angles, msg.ranges):
        for i, distance in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if np.isfinite(distance) and distance < msg.range_max:
                world_x = self.current_x + distance * np.cos(angle + self.current_heading)
                world_y = self.current_y + distance * np.sin(angle + self.current_heading)
                grid_x, grid_y = self.world_to_grid(world_x, world_y)
                
                # Inflate obstacles by WALL_MARGIN.
                inflation_radius = int(WALL_MARGIN / GRID_RESOLUTION)
                x_min = max(0, grid_x - inflation_radius)
                x_max = min(self.grid_width, grid_x + inflation_radius + 1)
                y_min = max(0, grid_y - inflation_radius)
                y_max = min(self.grid_height, grid_y + inflation_radius + 1)
                self.occupancy_grid[x_min:x_max, y_min:y_max] = False
        
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
        # Fallback to current position if no free space found.
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
                # Stop if we're close enough to the goal.
                if math.hypot(new_node.x - self.goal_x, new_node.y - self.goal_y) < GOAL_TOLERANCE:
                    break
        # Backtrack to extract the path.
        path = []
        if len(nodes) > 1:
            current_idx = len(nodes) - 1
            while current_idx != -1:
                path.append(nodes[current_idx])
                current_idx = nodes[current_idx].parent_idx
            path.reverse()
        return path

    def send_drive_command(self, target_x: float, target_y: float):
        """Publish a drive command based on a pure-pursuit controller with steering smoothing.
           If the heading error is large, blend the target with a default lookahead."""
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - self.current_heading
        # Normalize heading error to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        max_steering = 0.4
        new_steering = np.clip(heading_error, -max_steering, max_steering)
        
        # If the heading error is large, blend the target with a simple lookahead
        if abs(heading_error) > (math.pi / 6):  # threshold of 30 degrees
            lookahead_x = self.current_x + LOOKAHEAD_DISTANCE * math.cos(self.current_heading)
            lookahead_y = self.current_y + LOOKAHEAD_DISTANCE * math.sin(self.current_heading)
            # Blend the target with the lookahead (50/50 mix)
            target_x = (target_x + lookahead_x) / 2.0
            target_y = (target_y + lookahead_y) / 2.0
            # Recompute error based on blended target
            dx = target_x - self.current_x
            dy = target_y - self.current_y
            target_angle = math.atan2(dy, dx)
            heading_error = target_angle - self.current_heading
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            new_steering = np.clip(heading_error, -max_steering, max_steering)
        
        # Apply low-pass filtering to smooth the steering command.
        alpha = 0.7  # smoothing factor between 0 (no smoothing) and 1 (full new value)
        filtered_steering = alpha * new_steering + (1 - alpha) * self.prev_steering
        self.prev_steering = filtered_steering

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = filtered_steering

        base_speed = 2.0
        steering_factor = 1.0 - abs(heading_error) / (math.pi / 2)
        drive_msg.drive.speed = base_speed * steering_factor

        self.drive_pub.publish(drive_msg)

    def odom_callback(self, msg: Odometry):
        """Update vehicle state, update the goal based on current position, and plan/control a path."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        self.current_heading = math.atan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                                          1.0 - 2.0 * (quat.y**2 + quat.z**2))
        
        # --- Goal logic based on the current location ---
        # Right side of the loop
        if self.current_x <= 7.80 and self.current_y <= 2.34:
            self.goal_x = self.current_x + 2.30
            self.goal_y = -0.145
            self.x_limit_top = self.current_x + 2.50
            self.x_limit_bot = self.current_x
            self.y_limit_left = 0.37
            self.y_limit_right = -0.66
        # Top side of the loop
        elif self.current_x > 7.80 and self.current_y <= 6.15:
            self.goal_x = 9.575
            self.goal_y = self.current_y + 2.30
            self.x_limit_top = 10.03
            self.x_limit_bot = 9.12
            self.y_limit_left = self.current_y + 2.50
            self.y_limit_right = self.current_y
        # Left side of the loop
        elif self.current_x >= -11.40 and self.current_y > 6.15:
            self.goal_x = self.current_x - 2.30
            self.goal_y = 8.65
            self.x_limit_top = self.current_x
            self.x_limit_bot = self.current_x - 2.50
            self.y_limit_left = 9.15
            self.y_limit_right = 8.15
        # Bottom side of the loop
        elif self.current_x < -11.40 and self.current_y > 2.34:
            self.goal_x = -13.79
            self.goal_y = self.current_y - 2.30
            self.x_limit_top = -13.32
            self.x_limit_bot = -14.26
            self.y_limit_left = self.current_y
            self.y_limit_right = self.current_y - 2.50

        self.get_logger().debug(
            f"Pos: ({self.current_x:.2f}, {self.current_y:.2f}) | Goal: ({self.goal_x:.2f}, {self.goal_y:.2f})"
        )
        
        # Plan the path toward the computed global goal.
        path = self.plan_path()
        if path:
            # Instead of picking the first valid node, iterate over the entire path
            # and choose the node with minimal absolute heading error (while being at least LOOKAHEAD_DISTANCE away).
            best_node = None
            best_error = float('inf')
            forward_vec = np.array([np.cos(self.current_heading), np.sin(self.current_heading)])
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
                # Optionally log candidate nodes.
                self.get_logger().debug(f"Node: ({node.x:.2f},{node.y:.2f}) | dist: {dist:.2f} | error: {error:.2f}")
                if error < best_error:
                    best_error = error
                    best_node = node
            if best_node is None:
                best_node = path[-1]
            self.send_drive_command(best_node.x, best_node.y)
        else:
            # If no valid path is found, stop the vehicle.
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RRTPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
