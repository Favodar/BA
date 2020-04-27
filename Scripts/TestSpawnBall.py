import rospy
from gazebo_msgs.srv import  SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
from geometry_msgs.msg import *

print("Waiting for gazebo services...")
rospy.init_node("spawn_products_in_bins")
rospy.wait_for_service("gazebo/delete_model")
rospy.wait_for_service("gazebo/spawn_sdf_model")
print("Got it.")

delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

initial_pose = Pose()
initial_pose.position.x = 1.7
initial_pose.position.y = 1.0
initial_pose.position.z = 2.7

model = open("sphere.sdf","r")
sdf = model.read()
model_name = "ball2"
try:
    spawn_model(model_name, sdf, "", initial_pose, "world")
except:
    pass
model.close()