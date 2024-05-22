from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    
    talker_node = Node(
        package="lab1_pkg",
        executable="talker"
    )
    
    
    relay_node = Node(
        package="lab1_pkg",
        executable="relay"
    )
    
    ld.add_action(talker_node)
    ld.add_action(relay_node)
    return ld