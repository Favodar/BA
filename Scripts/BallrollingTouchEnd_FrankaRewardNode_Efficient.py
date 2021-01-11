import rospy
from std_msgs.msg import Float64
import math
import time
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetLinkState
from std_srvs.srv import Empty
from RosControlAdapter import RosControlAdapter
from gazebo_msgs.srv import  SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
import random
import os
from timeout import timeout

"""
A class that implements mid-level control over the virtual Franka Panda cobot via ROS and gazebo.
It is structured for use with reinforcement learning, therefore the control signals are sent in
the "getObservation" function which returns an observation about a state under the condition that
certain actions have been taken. In other words, it forces the whole process in a
state1(action) -> state2, reward
structure, so that calculations can be made which assume the process is an MDP.
"""
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
    ['unit_sphere', 'unit_sphere::link'],
    ['target_sphere', 'target_sphere::link']
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

    def __init__(self, action, signal_rate=100, signal_repetitions=25, randomBall = True, ballPos = None, randomTarget = False, targetPos = None, steps_after_kick = 50):
     
        self.steps_after_kick = steps_after_kick
        self.reset_actions = [0.0 for j in self.joint]
        self.isRandomBall = randomBall
        self.ballPos = ballPos
        self.isRandomTarget = randomTarget
        self.targetPos = targetPos
        self.hitJackpot1 = False
        self.hitJackpot2 = False
        self.hitJackpot3 = False
        self.movedBall = False
        self.hasEnded = False

        self.jackpot1 = 1000
        self.jackpot2 = 2000
        self.jackpot3 = 3000

        self.signal_rate = signal_rate
        self.signal_repetitions = signal_repetitions
        self.gazebo = RosControlAdapter()
        self.gazebo.initialize()
        self.gazebo.unpause()
        self.rate = 1
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.link_coordinates = rospy.ServiceProxy(
            '/gazebo/get_link_state', GetLinkState)
        self.spawn_model = rospy.ServiceProxy(
            "gazebo/spawn_sdf_model", SpawnModel)
        self.delete_model = rospy.ServiceProxy(
            "gazebo/delete_model", DeleteModel)
        model = open(
            "/home/ryuga/Documents/BA/SDFModels/unit_sphere_4cm_radius/model.sdf", "r")
        self.ball_model = model.read()
        model.close()


        # initialize 2D array and fill with zeros
        self.observations = [[0 for j in range(3)] for obj in self.observedObjects]
        i = 0
        self.rospyPublishers = [0 for i in self.joint]
        for joint_position in self.joint:
            self.rospyPublishers[i] = rospy.Publisher(self.joint[i], Float64, queue_size=10)
            i += 1

        self.spawnTarget()
        self.deleteBall()
        self.spawnBall()
        self.initial_distance = self.getDistance(self.link_coordinates(self.observedObjects[9][1], ""), self.link_coordinates(self.observedObjects[10][1], ""))
        self.last_ball_position = self.link_coordinates(self.observedObjects[9][1], "")



    def initializeNode(self):
        try:
            rospy.init_node('franka_publisher_node', anonymous=True)
            self.rate = rospy.Rate(self.signal_rate)
        except rospy.ROSInterruptException:
            print("couldnt Initialize Node!!! cant sleep then")
            pass

    def getObservation(self, action, reset = False, set_reset_steps_to_zero = False):
        #print("action = " + str(action))

        self.unpause()
        #print("unpause physics")

        if not reset:
            repetitions = self.signal_repetitions
        else: # give the robot 1.5 seconds to reset its position
            self.hasEnded = False
            if(set_reset_steps_to_zero):
                repetitions = 0
                print("resetting without reps")
            else:
                repetitions = int(self.signal_rate*3.5)
            self.delete_model = rospy.ServiceProxy(
                "gazebo/delete_model", DeleteModel)
            #print("try to delete and spawn ball")
            self.deleteBall()
            self.spawnBall()
            self.spawnTarget()
            self.hitJackpot1 = False
            self.hitJackpot2 = False
            self.hitJackpot3 = False
            self.movedBall = False
            self.initial_distance = self.getDistance(self.link_coordinates(self.observedObjects[9][1], ""), self.link_coordinates(self.observedObjects[10][1], ""))
        
        
        i = 0
        for x in range(repetitions):
            for joint_position in action:
                self.rospyPublishers[i].publish(joint_position)
                i += 1
            i = 0
            #print("if the program hangs and this is the latest message, rate.sleep() from rospy.Rate IS THE FUCKING PROBLEM")
            self.rate.sleep()
        
        self.pause()
        #print("pause physics")

        # You have no idea how long it took me to get this working
        try:
            
            i = 0
            for obj in self.observedObjects:
                coordinates = self.link_coordinates(obj[1], "")
                self.observations[i][0] = coordinates.link_state.pose.position.x
                self.observations[i][1] = coordinates.link_state.pose.position.y
                self.observations[i][2] = coordinates.link_state.pose.position.z
                i+=1

        except rospy.ServiceException as e:
            print("Retrieving observations failed!")
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))

        #print(self.observations)
        return self.observations

    """
    Returns a one-time reward if the ball was moved and continuous reward for closeness of the ball to the target (can be negative if the ball is way off target).
    Additional one-time rewards if the ball crosses certain radii around the target. This is supposed to reward coming close to the target even if the ball proceeds
    to roll far away afterwards.
    If reward is 0, distance measurement probably failed. Check if topic strings in observedObjects are correct, gazebo is properly initialised etc
    """
    def getReward(self):
            try:
                reward = 0

                coordinates_ball = self.link_coordinates(self.observedObjects[9][1], "")
                coordinates_target = self.link_coordinates(self.observedObjects[10][1], "") 
                coordinates_finger = self.link_coordinates(self.observedObjects[7][1], "")

                distance_ball_target = self.getDistance(coordinates_ball, coordinates_target)

                #if(distance_ball_target < 0.05):
                #    distance_ball_target = 0.05
                
                if not (self.initial_distance==distance_ball_target):
                    
                    if(self.movedBall == False):
                        print("Ball moved! +100")
                        self.movedBall = True
                        reward += 100
                        self.hasEnded = True
                        for step in range(self.steps_after_kick):
                            self.getObservation(self.reset_actions)
                            temp_reward, temp_ended = self.getReward()
                            reward += temp_reward
                        print("reward = " + str(reward))

                    reward +=10
                    reward -= (distance_ball_target-2)*5
                    #reward += 20/(distance_ball_target**2)-distance_ball_target
                    #reward += 1000*self.getDistance(self.last_ball_position, coordinates_ball)

                    if((distance_ball_target < 1.0)):
                        if(self.hitJackpot1 == False):
                            reward += self.jackpot1
                            self.hitJackpot1 = True
                            print("Close to the target! +" + str(self.jackpot1))

                        if((distance_ball_target < 0.5)):
                            if(self.hitJackpot2 == False):
                                reward += self.jackpot2
                                self.hitJackpot2 = True
                                print("Very close to the target! +" + str(self.jackpot2))

                            if((distance_ball_target < 0.1)):
                                if(self.hitJackpot3 == False):
                                    reward += self.jackpot3
                                    self.hitJackpot3 = True
                                    print("Hit the target! +" + str(self.jackpot3))

                self.last_ball_position = coordinates_ball

                return reward, self.hasEnded

            except rospy.ServiceException as e:
                rospy.loginfo("Get Model State service call failed:  {0}".format(e))

            return 0

    def getDistance(self, coordinates1, coordinates2):
                xdistance = coordinates1.link_state.pose.position.x - coordinates2.link_state.pose.position.x
                ydistance = coordinates1.link_state.pose.position.y - coordinates2.link_state.pose.position.y
                zdistance = coordinates1.link_state.pose.position.z - coordinates2.link_state.pose.position.z
                distance = math.sqrt(xdistance**2 + ydistance**2 + zdistance**2)
                
                return distance
    
    def spawnBall(self, name = "unit_sphere"):
        initial_pose = Pose()
        if(self.isRandomBall):
            
            sign1 = random.choice([1, -1])
            sign2 = random.choice([1, -1])
            initial_pose.position.x = sign1*random.uniform(0.3, 0.6)
            initial_pose.position.y = sign2*random.uniform(0.3, 0.6)
            initial_pose.position.z = 0
        else:
            if(self.ballPos == None):
                initial_pose.position.x = 0.5
                initial_pose.position.y = 0.5
                initial_pose.position.z = 0
            else:
                initial_pose.position.x = self.ballPos[0]
                initial_pose.position.y = self.ballPos[1]
                initial_pose.position.z = self.ballPos[2]
        try:
            self.spawn_model(name, self.ball_model, "", initial_pose, "world")
        except:
            print("spawning model failed. trying again.")
            try:
                self.spawn_model(name, self.ball_model,
                                 "", initial_pose, "world")
            except:
                pass
            pass

    def spawnTarget(self, name="target_sphere"):
        self.deleteBall(name)
        initial_pose = Pose()
        if(self.isRandomTarget):

            sign1 = random.choice([1, -1])
            #sign2 = random.choice([1, -1])
             
            #initial_pose.position.x = sign2*random.uniform(2.0, 3.0)
            #initial_pose.position.y = sign1*random.uniform(2.0, 3.0)
            if(sign1==1):
                initial_pose.position.x = 2.0
                initial_pose.position.y = -1.0
            else:
                initial_pose.position.x = -1.0
                initial_pose.position.y = 2.0

            initial_pose.position.z = 0
        else:
            if(self.targetPos == None):
                initial_pose.position.x = 2.0
                initial_pose.position.y = -1.0
                initial_pose.position.z = 0
            else:
                initial_pose.position.x = self.targetPos[0]
                initial_pose.position.y = self.targetPos[1]
                initial_pose.position.z = self.targetPos[2]
        try:
            self.spawn_model(name, self.ball_model, "", initial_pose, "world")
        except:
            print("spawning target model failed. trying again.")
            try:
                self.spawn_model(name, self.ball_model,
                                 "", initial_pose, "world")
            except:
                pass
            pass
        

    def deleteBall(self, name = "unit_sphere"):
        try:
            self.delete_model(name)
        except:
            pass



