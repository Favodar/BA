import numpy as np
from abc import ABC, abstractmethod
import graphics
import time


class Object2D:

    @abstractmethod
    def get2DpointList(self):
        pass



class Car():
    def __init__ (self, rotation_step_size = 0.01745*5, maxspeed = 1.0, borderless = False):
        """[summary]
        
        Keyword Arguments:
            number_of_joints {int} -- must be 1 or more (default: {2})
            motor_speed {float} -- motor_speed = 1 means a rotation of circa 1 degree per step (default: {1.0})
        """        
        self.rotation_step_size = rotation_step_size
        self.maxspeed = maxspeed
        self.borderless = borderless

        self.rotation = 0
        self.coordinates = np.array([[50.0, 50.0],])
        self.speed = 0

        print("Car initialized.")
        print("rotation_step_size = " + str(self.rotation_step_size))
        print("maxspeed = " + str(maxspeed))
        print("joint coordinates:" +str(self.coordinates))
    
    def move(self, gaspedal = 0, rotate = 0):

        if(gaspedal!=0):
            if(self.speed<self.maxspeed):
                self.speed+=(0.05*gaspedal)

        elif(self.speed>0):
            self.speed-=0.1
            
        if(rotate==0):
            self.rotation -= self.rotation_step_size
        elif(rotate==2):
            self.rotation += self.rotation_step_size
        elif(rotate!=1):
            print("invalid rotate value! Must be -1, 0 or 1!")

        self.coordinates[0][0] += self.speed*np.sin(self.rotation)
        self.coordinates[0][1] += self.speed*np.cos(self.rotation)
        
        if(self.borderless):
            self.coordinates[0][0] %= 100
            self.coordinates[0][1] %= 100

    def get2DpointList(self):
        return self.coordinates



class Robot():

    total_arm_length = 25.0


    def __init__ (self,  number_of_joints = 3, step_size = 0.01745, motor_speed = 1.0):
        """[summary]
        
        Keyword Arguments:
            number_of_joints {int} -- must be 1 or more (default: {2})
            motor_speed {float} -- motor_speed = 1 means a rotation of circa 1 degree per step (default: {1.0})
        """        
        self.number_of_joints = number_of_joints
        self.motor_speed = motor_speed
        self.step_size = step_size*motor_speed

        self.joints = np.zeros((number_of_joints,), float)
        self.joint_coordinates = np.zeros((number_of_joints, 2), float)

        self.joint_coordinates[0] = [50.0,50.0]

        self.refresh()

        print("Robot initialized.")
        print("number of joints: " + str(self.number_of_joints))
        print("step_size = " + str(step_size))
        print("motor_speed = " + str(motor_speed))
        print(str(step_size) + "*" + str(motor_speed) + "=" + str(step_size*motor_speed))
        print("rotation step size " + str(self.step_size) + " (0.01745 = 1 deg)")
        print("joint rotations:" + str(self.joints))
        print("joint coordinates:" +str(self.joint_coordinates))

  
    def moveArm(self, joint_id, target_rotation):    
        """
        Moves arm in the direction of target_rotation (rotation in radians)
        
        Arguments:
            joint_id {int} -- first joint is 0, second joint 1 etc.
            target_rotation {float} -- rotation in radians
        """        
        normalized_target_rotation = target_rotation%(2*np.pi)

        difference = normalized_target_rotation-self.joints[joint_id]

        if(abs(difference)<self.step_size):
            self.joints[joint_id] = normalized_target_rotation

        elif(difference<np.pi):
            self.joints[joint_id] += self.step_size
        else:
            self.joints[joint_id] -= self.step_size

        self.joints[joint_id] %= 2*np.pi

        self.refresh()

    def refresh(self):
        limb_length = self.total_arm_length/self.number_of_joints

        current_rotation = 0.0
        

        for i in range (self.number_of_joints-1):
            current_rotation += self.joints[i]
            self.joint_coordinates[i+1][0] = self.joint_coordinates[i][0] + np.sin(current_rotation)*limb_length
            self.joint_coordinates[i+1][1] = self.joint_coordinates[i][1] + np.cos(current_rotation)*limb_length

    def getJointCoordinates(self, joint_id, rotation, length):

        if(joint_id==0): return [50, 50]

        xy = self.getJointCoordinates(joint_id-1, rotation, length)
        return [xy[0] + np.sin(rotation*joint_id)*length, xy[1] + np.cos(rotation*joint_id)*length]

        

        
        

        

    def get2DpointList(self):
        return self.joint_coordinates

class Ball(Object2D):
    
    def __init__(self, xPos, yPos):
        self.coordinates = np.array([[xPos, yPos],])

    def get2DpointList(self):
        return self.coordinates

class Render:

    colors = ["green", "red", "blue", "black"]
    cnumber = 4

    def __init__ (self, object2DList):
        self.objectList = object2DList
        self.window1 = graphics.GraphWin("window1", 1000, 1000)
        self.window1.setCoords(0, 0, 100, 100)
        self.graphicsObjectList = []
        self.text = graphics.Text(graphics.Point(1, 50), str(""))

    def setObjects (self, object2DList):
        self.objectList = object2DList


    
    def renderFirstFrame(self, reward = 0):
        for gObj in self.graphicsObjectList:
            gObj.undraw()
        self.graphicsObjectList = []

        textpos = graphics.Point(50, 5)
        self.text.undraw()
        self.text = graphics.Text(textpos, str(reward))
        self.text.draw(self.window1)
        i = 0
        for obj in self.objectList:
            points = obj.get2DpointList()
            for point in points:
                p1 = graphics.Point(point[0], point[1])
                c = graphics.Circle(p1, 1)
                c.setFill(self.colors[i%self.cnumber])
                c.draw(self.window1)
                self.graphicsObjectList.append(c)
            i+=1

    def renderFrame(self, reward = 0):
        for gObj in self.graphicsObjectList:
            gObj.undraw()
        self.graphicsObjectList = []

        textpos = graphics.Point(50, 5)
        self.text.undraw()
        self.text = graphics.Text(textpos, str(reward))
        self.text.draw(self.window1)
        i = 0
        for obj in self.objectList:
            points = obj.get2DpointList()
            for point in points:
                p1 = graphics.Point(point[0], point[1])
                c = graphics.Circle(p1, 1)
                c.setFill(self.colors[i%self.cnumber])
                c.draw(self.window1)
                self.graphicsObjectList.append(c)
            i+=1

        #self.window1.getMouse() # Pause to view result
        # window.close()    # Close window when done

        


# myRobot = Robot(3)
# myBall = Ball(5,1)
# myRender = Render([myRobot, myBall])
# for i in range(1000):
#     myRobot.moveArm(0, 0.01*i)
#     myRobot.moveArm(1, 0.005*i)
#     myRender.renderFrame()
#     time.sleep(0.02)

# print("lets go")


        



            


    