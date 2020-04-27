import rospy
from std_msgs.msg import Float64
import math
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetLinkState

observedObjects = [

    ['panda', 'panda_link0'],
    ['panda', 'panda_link1'],
    ['panda', 'panda_link2'],
    ['panda', 'panda_link3'],
    ['panda', 'panda_link4'],
    ['panda', 'panda_link5'],
    ['panda', 'panda_link6'],
    ['panda', 'panda_rightfinger'],
    ['panda', 'panda_leftfinger'],
    ['unit_sphere', 'unit_sphere::link']
]
        # 'block_b': Block('panda', 'panda_leftfinger'),
        # 'block_b': Block('unit_sphere', 'link'),

# returns 10 divided by distance as reward, with a maximum of reward = 1000 and a minimum of reward>0. If reward is 0, distance measurement probably failed. Check if topic strings in observedObjects are correct, gazebo is properly initialised etc
def getReward():
        try:
            link_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            
            counter = 0
            for obj in observedObjects:
                #print(observedObjects[counter][1])
                print("coordinates" + str(counter) + " :" + str(link_coordinates(obj[1], "")))
                #print("coordinates" + str(counter) + " :" + str(model_coordinates(observedObjects[counter][0], observedObjects[counter][1])))
                counter+=1

            coordinates1 = link_coordinates(observedObjects[7][1], "")
            
            coordinates2 = link_coordinates(observedObjects[9][1], "")

            print("My Type is: " + str(type(coordinates1)))
            print("coordinates1.link_state.pose.position.x" + str(coordinates1.link_state.pose.position.x))

            print("coordinates1: " + str(coordinates1))
            print("coordinates2: " + str(coordinates2))
            #print 'coordinates1.success = ', coordinates1.success
            #print 'coordinates2.success = ', coordinates2.success
            xdistance = coordinates1.link_state.pose.position.x - coordinates2.link_state.pose.position.x
            ydistance = coordinates1.link_state.pose.position.y - coordinates2.link_state.pose.position.y
            zdistance = coordinates1.link_state.pose.position.z - coordinates2.link_state.pose.position.z
            
            distance = math.sqrt(xdistance**2 + ydistance**2 + zdistance**2)

            print ("distance = " + str(distance))
            

            if(distance<0.01):
                distance = 0.01

            reward = 10/distance

            print ("reward = " + str(reward))

            return reward

        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))

        return 0

getReward()