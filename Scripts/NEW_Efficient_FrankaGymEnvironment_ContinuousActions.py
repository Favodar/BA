import gym
from gym import spaces
import math
import numpy as np
from NEW_Efficient_FrankaGymRewardNode3RandomBall import GymReward

class CustomEnv(gym.Env):
  """ Custom Environment that implements OpenAI gym interface """

  metadata = {'render.modes': ['human']}

  step_counter = 0
  
  
  def __init__(self, signal_rate=100, signal_repetitions=25, step_limit=8, physics_stepsize=0.001, number_of_joints = 7, randomBall = True, ballPos = None):
    """
    Custom Environment that implements OpenAI gym interface

    :param signal_rate (int, optional): How often motor signals are sent to the virtual robot, per second. Defaults to 100.      
    :param signal_repetitions (int, optional): How often the same signal is repeated. If signal rate is 100 and repetitions is 25, one signal ("action") is repeated for 0.25 seconds. Defaults to 25.
    :param step_limit (int, optional): How many actions comprise an episode. Defaults to 8.
    :param physics_stepsize (float, optional): How much time between physics updates (e.g. collision detection). Bigger step size makes the simulation run faster, at the cost of accuracy. Big values can lead to the robot collapse into itself. Sensible values lie between 0.001 (default) and 0.01 (10x faster but deteriorating robot behaviour). Defaults to 0.001.

    Args:
        signal_rate (int, optional): How often motor signals are sent to the virtual robot, per second. Defaults to 100.
        signal_repetitions (int, optional): How often the same signal is repeated. If signal rate is 100 and repetitions is 25, one signal ("action") is repeated for 0.25 seconds. Defaults to 25.
        step_limit (int, optional): How many actions comprise an episode. Defaults to 8.
        physics_stepsize (float, optional): How much time between physics updates (e.g. collision detection). Bigger step size makes the simulation run faster, at the cost of accuracy. Big values can lead to the robot collapse into itself. Sensible values lie between 0.001 (default) and 0.01 (10x faster but deteriorating robot behaviour). Defaults to 0.001.
    """
    self.number_of_joints = number_of_joints
    self.actions = [0.0 for j in range(self.number_of_joints)]
    self.engine = GymReward(self.actions, signal_rate, signal_repetitions, randomBall = randomBall, ballPos = ballPos)
    self.step_limit = step_limit
    self.signal_rate = signal_rate
    self.signal_repetitions = signal_repetitions
    self.rotation_step = math.pi/self.step_limit

    self.action_space = spaces.Box(-math.pi, math.pi,
                                   shape=(self.number_of_joints,), dtype=np.float32)

    self.observation_space = spaces.Box(-np.inf,
                                        np.inf, shape=(10, 3,), dtype=np.float32)

    self.engine.initializeNode()


    # This does not work yet:
    # self.reward.gazebo.__time_step = physics_stepsize

    


  def step(self, action):

    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(
        action), "%r (%s) invalid" % (action, type(action))


        
        
    observation = self.engine.getObservation(action)
    reward = self.engine.getReward()

    self.step_counter += 1

    done = (self.step_counter>=self.step_limit)

    # info = "I don't know what 'info' is supposed to contain."

    return observation, reward, done, {} # info

  def reset(self):

    self.step_counter = 0

    for i in range(self.number_of_joints):
      self.actions[i] = 0

    print("reset actionlist = " + str(self.actions))

    observation = self.engine.getObservation(self.actions, True)
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    print ("The robot can be observed by opening the gazebo GUI")

  def close(self):
        print ("close() has been called")
