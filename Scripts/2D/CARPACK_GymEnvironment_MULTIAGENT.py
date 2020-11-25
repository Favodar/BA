import gym
from gym import spaces
import math
import time
import random
import numpy as np
from RobotSimulation2D_TRIAL import Car, Render, Ball

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  step_counter = 0
  episode_counter = 0
  
  def __init__(self, defender_step_limit, defender_step_size, defender_maxspeed, defender_acceleration, defender_binaryReward, isEscaping, enemy_model, attacker_step_limit, attacker_step_size, attacker_maxspeed, attacker_acceleration, attacker_binaryReward, number_of_attackers = 6, number_of_defenders = 1):

    self.number_of_attackers = number_of_attackers
    self.number_of_defenders = number_of_defenders

    self.attacker_cars = []
    self.defender_cars = []
    
    self.enemy_model = enemy_model
    self.defender_step_limit = defender_step_limit
    self.defender_step_size = defender_step_size
    self.defender_maxspeed = defender_maxspeed
    self.defender_acceleration = defender_acceleration
    self.defender_binaryReward = defender_binaryReward
    
    self.isEscaping = isEscaping

    self.attacker_step_limit = attacker_step_limit
    self.attacker_step_size = attacker_step_size
    self.attacker_maxspeed = attacker_maxspeed
    self.attacker_acceleration = attacker_acceleration
    self.attacker_binaryReward = attacker_binaryReward
    
    if(isEscaping):
        self.number_of_actions = number_of_defenders
    else:
        self.number_of_actions = number_of_attackers
    #self.number_of_cars = 1
    self.episodeIsOver = False

    self.attacker_positions = []
 
    self.defender_positions = []
    
    self.resetCars()

    self.myRender = Render(self.attacker_cars)

    # super(CustomEnv, self).__init__()

    actionlist = [3 for j in range(self.number_of_actions)]

    self.action_space = spaces.MultiDiscrete(actionlist)
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2+number_of_attackers,2), dtype=np.float32) #self.number_of_cars,
    self.observation = self.getObservation()



  def step(self, action):

    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    enemy_action, _states = self.enemy_model.predict(self.getObservation(isEnemy=True))
    self.enemy_car.move(enemy_action[0], enemy_action[1])
    self.defender_car.move(action[0], action[1])




    #self.renderSlow(50)



    self.observation = self.getObservation()
    
    
    #print(observation)
    reward = self.getReward(self.isEscaping)

    self.step_counter += 1

    # if(self.episode_counter%50==0):
    #     self.renderSlow(400)

    done = (self.episodeIsOver|(self.step_counter>=self.defender_step_limit))

    # info = "I don't know what 'info' is supposed to contain."

    return self.observation, reward, done, {} # info

  def getReward(self, isEscaping = False):
    
    if(isEscaping):

        reward = 0

        for acar in self.attacker_cars:
            for dcar in self.defender_cars:
        
                coordinates1 = acar.get2DpointList()[0]
                coordinates2 = dcar.get2DpointList()[0]

                xdistance = coordinates1[0] - coordinates2[0]
                ydistance = coordinates1[1] - coordinates2[1]                
                distance = math.sqrt(xdistance**2 + ydistance**2)

                if(self.defender_binaryReward):
                    if(distance<10):
                            self.episodeIsOver = True
                            return -self.defender_step_limit/2
                    else:
                        reward = 1
                
                else:
                    if(distance<0.01):
                        distance = 0.01

                    reward += math.sqrt(distance)-10/((distance/10)**2) #10/(distance/10)+(35-distance)#abs(35-distance)*(35-distance)
        return reward


    else:

        reward = 0

        for acar in self.attacker_cars:
            for dcar in self.defender_cars:
        
                coordinates1 = acar.get2DpointList()[0]
                coordinates2 = dcar.get2DpointList()[0]

                xdistance = coordinates1[0] - coordinates2[0]
                ydistance = coordinates1[1] - coordinates2[1]                
                distance = math.sqrt(xdistance**2 + ydistance**2)

                if(self.attacker_binaryReward):
                    if(distance<10):
                            self.episodeIsOver = True
                            return self.attacker_step_limit
                    else:
                        reward = -1
                
                else:
                    if(distance<0.01):
                        distance = 0.01

                    reward += 30/(distance/10)-(10+math.sqrt(distance))#abs(35-distance)*(35-distance)
        return reward

  def reset(self):
    self.render()

    self.step_counter = 0
    self.episodeIsOver = False
    self.episode_counter += 1

    self.resetCars()

    self.myRender.setObjects([self.defender_car, self.enemy_car])

    self.observation = self.getObservation()
    #observation = [self.myCar.coordinates[0][0], self.myCar.coordinates[0][1]]#, self.myCar.speed, self.myCar.rotation]


    return self.observation  # reward, done, info can't be included

  def render(self, mode='human'):
    self.myRender.renderFrame(self.getReward(self.isEscaping))

  def renderSlow(self, fps):
    self.myRender.renderFrame(self.getReward(self.isEscaping))
    time.sleep(1.0/fps)



  def close(self):
        print ("close() has been called")


  def getRandomPosition(self, role = "ATTACKER"):
    if(role=="ATTACKER"):

        sign1 = random.choice([1, -1])
        sign2 = random.choice([1, -1])
        x = 50 + sign1*random.randint(40, 50)
        y = 50 + sign2*random.randint(40, 50)

        rot =  random.randrange(0, 2*math.pi)
        
        
        return [x, y, rot]
    elif (role=="DEFENDER"):

        sign1 = random.choice([1, -1])
        sign2 = random.choice([1, -1])
        x = 50 + sign1*random.randint(0, 10)
        y = 50 + sign2*random.randint(0, 10)

        rot =  random.randrange(0, 2*math.pi)
        
        
        return [x, y, rot]

  def getObservation(self, isEnemy = False):
      if(isEnemy):
          return [[self.defender_car.coordinates[0][0], self.defender_car.coordinates[0][1]],[self.enemy_car.coordinates[0][0], self.enemy_car.coordinates[0][1]],[self.enemy_car.speed, self.enemy_car.rotation]]

      return [[self.enemy_car.coordinates[0][0], self.enemy_car.coordinates[0][1]],[self.defender_car.coordinates[0][0], self.defender_car.coordinates[0][1]],[self.defender_car.speed, self.defender_car.rotation]]

  def resetCars(self):

    self.attacker_cars = []
    self.defender_cars = []  
      
    self.attacker_positions = []
    for i in self.number_of_attackers:
        position = self.getRandomPosition()
        self.attacker_positions.append(position)

    self.defender_positions = []
    for i in self.number_of_defenders:
        position = self.getRandomPosition("DEFENDER")
        self.defender_positions.append(position)

    for position in self.attacker_positions:
        self.attacker_cars.append(Car(x_pos=position[0], y_pos=position[1], rotation=position[2], rotation_step_size=self.attacker_step_size, maxspeed=self.attacker_maxspeed, acceleration=self.attacker_acceleration))
    
    for position in self.defender_positions:
        self.defender_cars.append(Car(x_pos=position[0], y_pos=position[1], rotation=position[2], rotation_step_size=self.defender_step_size, maxspeed=self.defender_maxspeed, acceleration=self.defender_acceleration))