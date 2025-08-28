#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Package directory
    pkg_share = FindPackageShare('particle_filter_localization')
    
    # Declare arguments
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=PathJoinSubstitution([pkg_share, 'maps', 'closed_walls_map.yaml']),  # Replace 'map.yaml' with YOUR map filename
        description='Path to the map file'
    )
    
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=PathJoinSubstitution([pkg_share, 'worlds', 'closed_walls.world']),  # Replace 'world.world' with YOUR world filename
        description='Path to the world file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Map server node
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': LaunchConfiguration('map_file'),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    # Lifecycle manager for map server
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'node_names': ['map_server'],
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'autostart': True
        }]
    )
    
    # Your particle filter node
    particle_filter_node = Node(
        package='particle_filter_localization',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    # Include your four wheel robot simulation
    four_wheel_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_share,
                'launch',
                'gazebo_4_wheel.launch.py'
            ])
        ])
    )
    
    return LaunchDescription([
        map_file_arg,
        world_file_arg,
        use_sim_time_arg,
        four_wheel_launch,      # Launch your robot simulation first
        map_server_node,
        lifecycle_manager_node,
        particle_filter_node,
    ])