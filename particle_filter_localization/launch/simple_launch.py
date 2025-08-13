#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Just your particle filter node for now
    particle_filter_node = Node(
        package='particle_filter_localization',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        parameters=[{
            'use_sim_time': True  # Change to False if using real robot
        }]
    )
    
    return LaunchDescription([
        particle_filter_node,
    ])