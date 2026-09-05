import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'pennair_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        #ensure the file under launch is copied to share/pennair_vision/launch
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='PennAir Vision ROS 2 Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = pennair_vision.camera_node:main',
            'detection_node = pennair_vision.detection_node:main',
        ],
    },
)
