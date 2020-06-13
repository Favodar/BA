import gym
from gym import spaces
import math
import numpy as np
from NEW_FrankaGymRewardNode3RandomBall import GymReward
from timeout import timeout
import os
import time

from gazebo_msgs.srv import GetPhysicsProperties, GetWorldProperties, SetPhysicsProperties
from std_srvs.srv import Empty
import rospy

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  step_counter = 0
  
  number_of_joints = 7
  
  def __init__(self, signal_rate = 100, signal_repetitions = 25, step_limit = 8, physics_stepsize = 0.001):

    self.reward = GymReward(signal_rate, signal_repetitions)
    self.step_limit = step_limit
    self.signal_rate = signal_rate
    self.signal_repetitions = signal_repetitions

    # super(CustomEnv, self).__init__()

    actionlist = [3 for j in range(self.number_of_joints)]

    self.action_space = spaces.MultiDiscrete(actionlist)

    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,3,), dtype=np.float32)

    self.reward.initializeNode()

    # This does not work yet:
    # self.reward.gazebo.__time_step = physics_stepsize

    self.actions = [0.0 for j in range(self.number_of_joints)]

  # Wrapper class that restarts the environment if the environment hangs up during observation collection (does not work yet)
  def step(self, action):

    try:
        return self.step4real(action)
    except:
    #     self.reward.gazebo.reset()
    #     return self.step(action)

    # if(False):
        print("step() is not responding, killing ros and gazebo...")

        #self.reward.gazebo.shutdown()
        #time.sleep(1.0)
        stream = os.popen('killall gzclient')
        time.sleep(1.0)
        os.popen("killall gzserver")
        time.sleep(3.0)
        #os.popen("killall roscore")
        #time.sleep(1.0)
        os.popen("killall roslaunch")
        time.sleep(1.0)
        os.popen("killall rosmaster")
        time.sleep(2.0)

        print("restarting roscore and gazebo")

        os.popen("roscore")
        time.sleep(2.0)
        os.popen("roslaunch gazebo_ros empty_world.launch")
        time.sleep(3.0)
        os.popen("roslaunch franka_gazebo panda_arm_hand.launch")
        time.sleep(5.0)

        self.reward = GymReward(self.signal_rate, self.signal_repetitions)
        #rospy.wait_for_service('/gazebo/get_physics_properties')
        self.reward.gazebo.__get_physics_properties = rospy.ServiceProxy(
            'gazebo/get_physics_properties', GetPhysicsProperties, persistent=True)
        #rospy.wait_for_service('/gazebo/get_world_properties')
        self.reward.gazebo.__get_world_properties = rospy.ServiceProxy(
            'gazebo/get_world_properties', GetWorldProperties, persistent=True)
        #rospy.wait_for_service('/gazebo/set_physics_properties')
        self.reward.gazebo.__set_physics_properties = rospy.ServiceProxy(
            'gazebo/set_physics_properties', SetPhysicsProperties, persistent=True)
        #rospy.wait_for_service('/gazebo/pause_physics')
        self.reward.gazebo.__pause_client = rospy.ServiceProxy(
            'gazebo/pause_physics', Empty, persistent=True)
        #rospy.wait_for_service('/gazebo/unpause_physics')
        self.reward.gazebo.__unpause_client = rospy.ServiceProxy(
            'gazebo/unpause_physics', Empty, persistent=True)
        #rospy.wait_for_service('/gazebo/reset_sim')
        self.reward.gazebo.__reset = rospy.ServiceProxy(
            'gazebo/reset_sim', Empty, persistent=True)
        #rospy.wait_for_service('gazebo/end_world')
        self.reward.gazebo.__endWorld = rospy.ServiceProxy(
            'gazebo/end_world', Empty, persistent=True)
        self.reward.gazebo.__time_step = 0.001
        self.reward.gazebo.__is_initialized = False

        self.reward.gazebo.initialize()
        #self.reward.gazebo.reset()
        time.sleep(1.0)
        self.reward.initializeNode()
        self.reward.rate.last_time = rospy.rostime.get_rostime()

        return self.step(action)
        

  @timeout(15)
  def step4real(self, action):

    assert self.action_space.contains(
        action), "%r (%s) invalid" % (action, type(action))

    for i in range(self.number_of_joints):
        if(action[i] == 0):
            self.actions[i] -= math.pi/self.step_limit
        elif(action[i] == 1):
            self.actions[i] += 0
        elif(action[i] == 2):
            self.actions[i] += math.pi/self.step_limit
        else:
            print("unknown direction in step function: " + str(action[i]))

    observation = self.reward.getObservation(self.actions)
    reward = self.reward.getReward()

    self.step_counter += 1

    done = (self.step_counter >= self.step_limit)

    # info = "I don't know what 'info' is supposed to contain."

    return observation, reward, done, {}  # info

  def reset(self):

    self.step_counter = 0

    for i in range(self.number_of_joints):
      self.actions[i] = 0

    print("reset actionlist = " + str(self.actions))

    observation = self.reward.getObservation(self.actions, True)
    return observation  # reward, done, info can't be included

  def render(self, mode='human'):
    print ("The robot can be observed by opening the gazebo GUI")

  def close(self):
        print ("close() has been called")
