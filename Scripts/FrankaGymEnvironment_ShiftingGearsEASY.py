import gym
from gym import spaces
import math
import numpy as np
from FrankaGymRewardNode3RandomBallEASY import GymReward

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  step_counter = 0

  gear_interval = 0.05
  
  def __init__(self, signal_rate = 100, signal_repetitions = 25, step_limit = 8, controlled_joints = 2, number_of_gears = 1, gear_interval = 0.05):

    self.reward = GymReward(signal_rate, signal_repetitions)
    self.step_limit = step_limit
    self.number_of_joints = controlled_joints
    self.number_of_gears = number_of_gears
    self.gear_interval = gear_interval

    print("initializing agent...")
    print("number of gears: " + str(self.number_of_gears))
    print("gear interval: " + str(self.gear_interval))
    print("full throttle: " + str(self.number_of_gears*self.gear_interval))

    # super(CustomEnv, self).__init__()

    actionlist = [3 for j in range(self.number_of_joints)]

    self.action_space = spaces.MultiDiscrete(actionlist)

    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,3,), dtype=np.float32)

    self.reward.initializeNode()

    self.actions = [0.0 for j in range(self.number_of_joints)]
    self.gears = [0 for j in range(self.number_of_joints)]


  def step(self, action):

    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    print("step function action parameter:" + str(action))

    for i in range(self.number_of_joints):
        if(action[i]==0):
            if(self.gears[i]>(self.number_of_gears*(-1))):
                self.gears[i] -= 1
            self.actions[i] += self.gear_interval*self.gears[i]
        elif(action[i]==1):
            self.actions[i] += self.gear_interval*self.gears[i]
        elif(action[i]==2):
             if(self.gears[i]<self.number_of_gears):
                self.gears[i] += 1
             self.actions[i] += self.gear_interval*self.gears[i]
        else:
            print("unknown direction in step function: " + str(action[i]))

        
        
    observation = self.reward.getObservation(self.actions)
    reward = self.reward.getReward()

    self.step_counter += 1

    done = (self.step_counter>=self.step_limit)

    # info = "I don't know what 'info' is supposed to contain."

    return observation, reward, done, {} # info

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