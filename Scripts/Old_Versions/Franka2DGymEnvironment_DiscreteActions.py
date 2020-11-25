import gym
from gym import spaces
import math
import numpy as np
from Franka2DGymRewardNodeNOREPS import GymReward

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  step_counter = 0
  
  def __init__(self, number_of_joints = 3, step_limit = 1000, step_size = 0.01745):

    self.reward = GymReward(number_of_joints= number_of_joints, step_size = step_size)
    self.step_limit = step_limit
    self.step_size = step_size

    self.number_of_actions = number_of_joints - 1

    # super(CustomEnv, self).__init__()

    actionlist = [3 for j in range(self.number_of_actions)]

    self.action_space = spaces.MultiDiscrete(actionlist)

    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(number_of_joints+1,2,), dtype=np.float32)


    self.actions = [0.0 for j in range(self.number_of_actions)]


  def step(self, action):

    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    for i in range(self.number_of_actions):
        if(action[i]==0):
            self.actions[i] -= self.step_size
        elif(action[i]==1):
            self.actions[i] += 0
        elif(action[i]==2):
            self.actions[i] += self.step_size
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

    for i in range(self.number_of_actions):
      self.actions[i] = 0

    print("reset actionlist = " + str(self.actions))

    observation = self.reward.getObservation(self.actions, True)
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    self.reward.render()

  def close(self):
        print ("close() has been called")