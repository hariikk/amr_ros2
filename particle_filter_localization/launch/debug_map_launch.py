
#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Package directory
    pkg_share = FindPackageShare('particle_filter_localization')
    
    # Map server node with explicit path
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': PathJoinSubstitution([pkg_share, 'maps', 'closed_walls_map.yaml']),
            'use_sim_time': True
        }]
    )
    
    # Lifecycle manager
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'node_names': ['map_server'],
            'use_sim_time': True,
            'autostart': True
        }]
    )
    
    return LaunchDescription([
        map_server_node,
        lifecycle_manager_node,
    ])