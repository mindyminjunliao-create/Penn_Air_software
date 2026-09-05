from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start camera node
        Node(
            package='pennair_vision',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),
        # Start 3D pose detection node
        Node(
            package='pennair_vision',
            executable='detection_node',
            name='detection_node',
            output='screen'
        ),
    ])
