import rospy
from gazebo_msgs.srv import  SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
from geometry_msgs.msg import *


delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)


model_name = "unit_sphere"
try:
    delete_model(model_name)
except:
    pass
