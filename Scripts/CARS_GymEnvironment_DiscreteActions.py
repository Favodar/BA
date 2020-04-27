import gym
from gym import spaces
import math
import time
import random
import numpy as np
from RobotSimulation2D import Car, Render, Ball

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  step_counter = 0
  episode_counter = 0
  
  def __init__(self, step_limit, step_size, maxspeed, randomBall = False, binaryReward = False):

    self.step_limit = step_limit
    self.step_size = step_size
    self.maxspeed = maxspeed
    self.randomBall = randomBall
    self.binaryReward = binaryReward

    self.number_of_actions = 2
    self.number_of_cars = 1
    self.episodeIsOver = False

    self.myCar = Car(rotation_step_size = step_size, maxspeed= maxspeed)
    if(randomBall):
      self.spawnBall()
    else:
      self.myBall = Ball(75, 75)

    self.myRender = Render([self.myCar, self.myBall])

    # super(CustomEnv, self).__init__()

    actionlist = [3 for j in range(self.number_of_actions)]

    self.action_space = spaces.MultiDiscrete(actionlist)
    if(randomBall):
      self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,2), dtype=np.float32) #self.number_of_cars,
    else:
      self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,2), dtype=np.float32) #self.number_of_cars,
      



  def step(self, action):

    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    self.myCar.move(action[0], action[1])

    #self.renderSlow(400)



    observation = self.getObservation()
    
    #print(observation)
    reward = self.getReward()

    self.step_counter += 1

    # if(self.episode_counter%50==0):
    #     self.renderSlow(400)

    done = (self.episodeIsOver|(self.step_counter>=self.step_limit))

    # info = "I don't know what 'info' is supposed to contain."

    return observation, reward, done, {} # info

  def getReward(self):



        coordinates1 = self.myCar.get2DpointList()[0]
        coordinates2 = self.myBall.get2DpointList()[0]

        xdistance = coordinates1[0] - coordinates2[0]
        ydistance = coordinates1[1] - coordinates2[1]                
        distance = math.sqrt(xdistance**2 + ydistance**2)

        #print ("distance = " + str(distance))

        if(self.binaryReward):
          if(distance<5):
            self.episodeIsOver = True
            return 1
          else:
            return 0
        
        if(distance<0.01):
            distance = 0.01

        reward = 10/(distance/10)+(35-distance)#abs(35-distance)*(35-distance)

        #print ("reward = " + str(reward))

        return reward

  def reset(self):
    self.render()

    self.step_counter = 0
    self.episodeIsOver = False
    self.episode_counter += 1

    self.myCar = Car(rotation_step_size = self.step_size, maxspeed= self.maxspeed)
    if(self.randomBall):
      self.spawnBall()

    self.myRender.setObjects([self.myCar, self.myBall])

    observation = self.getObservation()
    #observation = [self.myCar.coordinates[0][0], self.myCar.coordinates[0][1]]#, self.myCar.speed, self.myCar.rotation]


    return observation  # reward, done, info can't be included

  def render(self, mode='human'):
    self.myRender.renderFrame(self.getReward())

  def renderSlow(self, fps):
    self.myRender.renderFrame(self.getReward())
    time.sleep(1.0/fps)



  def close(self):
        print ("close() has been called")


  def spawnBall(self):

        sign1 = random.choice([1, -1])
        sign2 = random.choice([1, -1])
        x = 50 + sign1*random.randint(40, 50)
        y = 50 + sign2*random.randint(40, 50)
        
        
        self.myBall = Ball(x, y)

  def getObservation(self):

    if(self.randomBall):
      return [[self.myBall.coordinates[0][0], self.myBall.coordinates[0][1]],[self.myCar.coordinates[0][0], self.myCar.coordinates[0][1]],[self.myCar.speed, self.myCar.rotation]]
    else:
      return [[self.myCar.coordinates[0][0], self.myCar.coordinates[0][1]],[self.myCar.speed, self.myCar.rotation]]