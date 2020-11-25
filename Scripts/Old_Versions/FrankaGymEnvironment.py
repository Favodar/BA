import gym
from gym import spaces
import math
import numpy as np
from FrankaGymRewardNode3RandomBall import GymReward

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  step_counter = 0
  
  number_of_actions = 7
  
  def __init__(self, signal_rate = 100, signal_repetitions = 25, step_limit = 8):

    self.reward = GymReward(signal_rate, signal_repetitions)
    self.step_limit = step_limit

    # super(CustomEnv, self).__init__()

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Box(-math.pi, math.pi, shape=(self.number_of_actions,), dtype= np.float32)
    # Example for using image as input:
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,3,), dtype=np.float32)

    

    self.reward.initializeNode()

  def step(self, action):

    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    observation = self.reward.getObservation(action)
    reward = self.reward.getReward()

    self.step_counter += 1

    done = (self.step_counter>=self.step_limit)

    # info = "I don't know what 'info' is supposed to contain."

    return observation, reward, done, {} # info

  def reset(self):

    self.step_counter = 0

    actionlist = []
    for i in range(self.number_of_actions):
      actionlist.append(0)

    print("reset actionlist = " + str(actionlist))

    observation = self.reward.getObservation(actionlist, True)
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    print ("The robot can be observed by opening the gazebo GUI")

  def close(self):
        print ("close() has been called")