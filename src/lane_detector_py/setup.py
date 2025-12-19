from setuptools import setup, find_packages

package_name = 'lane_detector_py'  # ROS 패키지명

setup(
    name=package_name,             # 배포 이름(대시/언더스코어 상관없지만 보통 동일)
    version='0.1.0',
    packages=find_packages(include=[package_name + '*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/lane_detector.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='ROS2 lane detection (rclpy + OpenCV)',
    license='MIT',
    entry_points={
        'console_scripts': [
            # 모듈 경로 **폴더명과 동일**해야 함
            'lane_detector_node = lane_detector_py.lane_detector_node:main',
        ],
    },
)
