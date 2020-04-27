import gym
from gym import spaces
import math
import numpy as np
from FrankaGymRewardNode3RandomBall import GymReward

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  step_counter = 0
  
  number_of_joints = 7
  
  def __init__(self, signal_rate = 100, signal_repetitions = 25, step_limit = 8):

    self.reward = GymReward(signal_rate, signal_repetitions)
    self.step_limit = step_limit

    # super(CustomEnv, self).__init__()

    actionlist = [3 for j in range(self.number_of_joints)]

    self.action_space = spaces.MultiDiscrete(actionlist)

    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,3,), dtype=np.float32)

    self.reward.initializeNode()

    self.actions = [0.0 for j in range(self.number_of_joints)]


  def step(self, action):

    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    for i in range(self.number_of_joints):
        if(action[i]==0):
            self.actions[i] -= 0.05
        elif(action[i]==1):
            self.actions[i] += 0
        elif(action[i]==2):
            self.actions[i] += 0.05
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