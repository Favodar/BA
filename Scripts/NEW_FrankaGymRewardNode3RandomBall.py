import rospy
from std_msgs.msg import Float64
import math
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetLinkState
from std_srvs.srv import Empty
from RosControlAdapter import RosControlAdapter
from gazebo_msgs.srv import  SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
import random

class GymReward:

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

    joint = [
    '/franka/joint1_position_controller/command',
    '/franka/joint2_position_controller/command',
    '/franka/joint3_position_controller/command',
    '/franka/joint4_position_controller/command',
    '/franka/joint5_position_controller/command',
    '/franka/joint6_position_controller/command',
    '/franka/joint7_position_controller/command',
    ]

    def __init__ (self, signal_rate = 100, signal_repetitions = 25):
        self.signal_rate = signal_rate
        self.signal_repetitions = signal_repetitions
        self.gazebo = RosControlAdapter()
        self.gazebo.initialize()
        self.gazebo.unpause()

    def initializeNode(self):
        try:
            rospy.init_node('franka_publisher_node', anonymous=True)
            # rate = rospy.Rate(self.signal_rate)
            # while not rospy.is_shutdown():
            #     print(self.__actions__[0])
            #     i = 0
            #     for joint_position in self.__actions__:
            #         pub = rospy.Publisher(self.joint[i], Float64, queue_size=10)
            #         pub.publish(joint_position)
            #         i += 1
            #     rate.sleep()
        except rospy.ROSInterruptException:
            pass



    def getObservation(self, action, reset = False):

        #self.gazebo.unpause()

        #print("action = " + str(action))
        i = 0
        rate = rospy.Rate(self.signal_rate)

        if not reset:
            repetitions = self.signal_repetitions
        else: # give the robot 1.5 seconds to reset its position
            repetitions = int(self.signal_rate*1.5)
            self.deleteBall()
            self.spawnBall()
        
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause()
        #print("unpause physics")

        for x in range(repetitions):
            for joint_position in action:
                pub = rospy.Publisher(self.joint[i], Float64, queue_size=10)
                pub.publish(joint_position)
                i += 1
            i = 0
            #print("rate.sleep()")
            rate.sleep()

        #self.gazebo.pause()
        
        pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        pause()

        #print("pause physics")


        # initialize 2D array and fill with zeros
        observations = [[0 for j in range(3)] for obj in self.observedObjects]

        try:
            link_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            i = 0
            for obj in self.observedObjects:
                coordinates = link_coordinates(obj[1], "")
                #observations.append((coordinates.pose.position.x, coordinates.pose.position.y, coordinates.pose.position.z))
                observations[i][0] = coordinates.link_state.pose.position.x
                observations[i][1] = coordinates.link_state.pose.position.y
                observations[i][2] = coordinates.link_state.pose.position.z

                i+=1


        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))

        #print(observations)
        return observations



    # returns 10 divided by distance as reward, with a maximum of reward = 1000 and a minimum of reward>0. If reward is 0, distance measurement probably failed. Check if topic strings in observedObjects are correct, gazebo is properly initialised etc
    def getReward(self):
            try:
                link_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
                
                coordinates1 = link_coordinates(self.observedObjects[7][1], "")
                coordinates2 = link_coordinates(self.observedObjects[9][1], "")

                #print("coordinates1: " + str(coordinates1))
                #print("coordinates2: " + str(coordinates2))
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

    def spawnBall(self, name = "unit_sphere"):

        spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

        initial_pose = Pose()
        sign1 = random.choice([1, -1])
        sign2 = random.choice([1, -1])
        initial_pose.position.x = sign1*random.uniform(1.0, 2.0)
        initial_pose.position.y = sign2*random.uniform(1.0, 2.0)
        initial_pose.position.z = 0

        model = open("/home/ryuga/Documents/BA/SDFModels/unit_sphere_4cm_radius/model.sdf","r")
        sdf = model.read()
        try:
            spawn_model(name, sdf, "", initial_pose, "world")
        except:
            pass
        model.close()

    def deleteBall(self, name = "unit_sphere"):
        delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        try:
            delete_model(name)
        except:
            pass



