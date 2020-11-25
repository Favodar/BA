from RobotSimulation2D_Revisited import Robot, Render, Ball
import random
import time
import math

class GymReward:

    observedObjects = []

    joint = [
        '/franka/joint1_position_controller/command',
    ]

    def __init__ (self, number_of_joints = 3, step_size = 0.01745):

        self.number_of_joints = number_of_joints
        self.step_size = step_size

        self.myRobot = Robot(number_of_joints = number_of_joints, step_size = step_size)
        self.spawnBall()

        self.myRender = Render([self.myRobot, self.myBall])
        
        # unpause





    def getObservation(self, action, reset = False):

        #self.gazebo.unpause()

        #print("action = " + str(action))
        

        if reset:
            self.renderFast()
            print("reward" + str(self.getReward()))
            self.myRobot = Robot(number_of_joints= self.number_of_joints, step_size = self.step_size)
            self.spawnBall()
            self.myRender.setObjects([self.myRobot, self.myBall])

        i = 0
        
        for joint_position in action:
            self.myRobot.moveArm(i, joint_position)
            i += 1
            

        #self.render()

        observedObjects = []

        for point in self.myRobot.get2DpointList():
            observedObjects.append(point)
        for point in self.myBall.get2DpointList():
            observedObjects.append(point)

        #print("observation: " + str(observedObjects))
        return observedObjects



    # returns 10 divided by distance as reward, with a maximum of reward = 1000 and a minimum of reward>0. If reward is 0, distance measurement probably failed. Check if topic strings in observedObjects are correct, gazebo is properly initialised etc
    def getReward(self):

        coordinates1 = self.myRobot.get2DpointList()[self.number_of_joints-1]
        coordinates2 = self.myBall.get2DpointList()[0]

        xdistance = coordinates1[0] - coordinates2[0]
        ydistance = coordinates1[1] - coordinates2[1]                
        distance = math.sqrt(xdistance**2 + ydistance**2)

        #print ("distance = " + str(distance))
        
        if(distance<0.01):
            distance = 0.01

        reward = 100-distance

        #print ("reward = " + str(reward))

        return reward

    def spawnBall(self):

        sign1 = random.choice([1, -1])
        sign2 = random.choice([1, -1])
        x = 50 + sign1*random.randint(40, 50)
        y = 50 + sign2*random.randint(40, 50)
        
        
        self.myBall = Ball(x, y)

    def spawnBallWithCoordinates(self, x, y):
        self.myBall = Ball(x,y)

    def deleteBall(self, name = "unit_sphere"):
        print("implement delete ball")

    def render(self, fps=100):
        self.myRender.renderFrame(reward=self.getReward())
        time.sleep(1.0/fps)

    def renderFast(self):
        self.myRender.renderFrame()




