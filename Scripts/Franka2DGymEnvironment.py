import gym
from gym import spaces
import math
import numpy as np
from Franka2DGymRewardNode import GymReward

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface

  This "easy" version has a smaller action and a smaller
  observation space (two actions and two 3D-observations).
  This is supposed to make it easier for the RL agent.
  """
  metadata = {'render.modes': ['human']}

  # how often the step function has been called in the current episode. Is used to determine whether an episode has ended.
  step_counter = 0
  

  
  def __init__(self, signal_repetitions = 25, number_of_joints = 3, step_limit = 100):

    self.reward = GymReward(signal_repetitions, number_of_joints)
    self.step_limit = step_limit

    self.number_of_actions = number_of_joints - 1

    # super(CustomEnv, self).__init__()

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Box(-math.pi, math.pi, shape=(self.number_of_actions,), dtype= np.float32)
    # Example for using image as input:
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(number_of_joints+1,2,), dtype=np.float32)

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

    #print("reset actionlist = " + str(actionlist))

    observation = self.reward.getObservation(actionlist, True)
    return observation  # reward, done, info can't be included

  def render(self, mode='human'):
    self.reward.render()

  def close(self):
        print ("close() has been called")