# 导入库
import os
from pickle import TRUE

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory

# 定义函数名称为：generate_launch_description
def generate_launch_description():
	# dsfbot_bringup_dir = get_package_share_directory('capella_ros_launcher')
	fast_lio_dir = get_package_share_directory('fast_lio')

	default_config_path = os.path.join(fast_lio_dir, 'config', 'mid360.yaml')
	rviz_config_dir = os.path.join(fast_lio_dir, 'rviz_cfg', 'loam_livox.rviz')

	config_path = LaunchConfiguration('config_path')

	declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    	)

    	#laserMapping node
	laserMapping_node = Node(
        package = "fast_lio",
        executable = "laserMapping_node",
        output = 'screen',  #四个可选项 
        parameters = [config_path]
        )

	rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        output='screen'
        )

	launch_description = LaunchDescription()

	launch_description.add_action(declare_config_path_cmd)

	launch_description.add_action(laserMapping_node)
	launch_description.add_action(rviz_node)

	return launch_description