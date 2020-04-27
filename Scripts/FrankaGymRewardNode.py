import rospy
from std_msgs.msg import Float64
import math
from gazebo_msgs.srv import GetModelState

class PublisherNode:

    # def __init__(self):
    #     rospy.init_node('franka_reward_node', anonymous=True)

    def initializeNode(self):
        rospy.init_node('franka_publisher_node', anonymous=True)
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
             rate.sleep()


    joint = [
    '/franka/joint1_position_controller/command',
    '/franka/joint2_position_controller/command',
    '/franka/joint3_position_controller/command',
    '/franka/joint4_position_controller/command',
    '/franka/joint5_position_controller/command',
    '/franka/joint6_position_controller/command',
    '/franka/joint7_position_controller/command',
    ]

    observedObjects = [
        ['panda', 'panda_rightfinger'],
        ['panda', 'panda_link0'],
        ['panda', 'panda_link1'],
        ['panda', 'panda_link2'],
        ['panda', 'panda_link3'],
        ['panda', 'panda_link4'],
    ]
        # 'block_b': Block('panda', 'panda_leftfinger'),
        # 'block_b': Block('unit_sphere', 'link'),



    def getObservation(self, action):
        i = 0
        for joint_position in action:
            pub = rospy.Publisher(self.joint[i], Float64, queue_size=10)
            pub.publish(joint_position)
            i += 1

        observations = []

        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            
            for obj in self.observedObjects:
                coordinates = model_coordinates(obj[0], obj[1])
                observations.append((coordinates.pose.position.x, coordinates.pose.position.y, coordinates.pose.position.z))


        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))

        return observations



    # returns 10 divided by distance as reward with a maximum of 100 and a minimum of >0. If reward is 0, distance measurement probably failed. Check if topic strings in observedObjects are correct, gazebo is properly initialised etc
    def getReward(self):
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

            coordinates1 = model_coordinates(self.observedObjects[0][0], self.observedObjects[0][1])
            coordinates2 = model_coordinates(self.observedObjects[1][0], self.observedObjects[1][1])
            #print 'coordinates1.success = ', coordinates1.success
            #print 'coordinates2.success = ', coordinates2.success
            xdistance = coordinates1.pose.position.x - coordinates2.pose.position.x
            ydistance = coordinates1.pose.position.y - coordinates2.pose.position.y
            zdistance = coordinates1.pose.position.z - coordinates2.pose.position.z
            distance = math.sqrt(xdistance**2 + ydistance**2 + zdistance**2)

            print ("distance = " + str(distance))
            

            if(distance<0.1):
                distance = 0.1

            reward = 10/distance

            # print "reward = " + str(reward)

            return reward

        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))

        return 0



