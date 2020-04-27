import os
import subprocess

""" os.system('rostopic pub /franka/joint1_position_controller/command std_msgs/Float64 "data: 0.5"')
os.system('rostopic pub /franka/joint2_position_controller/command std_msgs/Float64 "data: 0.5"')
os.system('rostopic pub /franka/joint3_position_controller/command std_msgs/Float64 "data: 0.5"')
subprocess.call(['rostopic', 'pub', '/franka/joint1_position_controller/command', 'std_msgs/Float64', '"data:', '0.5"'])
"""
# subprocess.run(['source', '/opt/ros/melodic/setup.bash'])
#subprocess.run(['rostopic', 'pub', '/franka/joint1_position_controller/command', 'std_msgs/Float64', '0.5'])
#subprocess.run(['^C'])
#subprocess.run(['rostopic', 'pub', '/franka/joint2_position_controller/command', 'std_msgs/Float64', '1.5'])
#subprocess.run(['^C'])
subprocess.run(['rostopic', 'pub', '/franka/joint3_position_controller/command', 'std_msgs/Float64', '2.5'])
subprocess.run(['^C'])
# subprocess.call(['rostopic', 'pub', '/franka/joint1_position_controller/command', 'std_msgs/Float64', '0.5'])
#subprocess.call('rostopic', 'pub', '/franka/joint1_position_controller/command', 'std_msgs/Float64', ['"0.5"']) 