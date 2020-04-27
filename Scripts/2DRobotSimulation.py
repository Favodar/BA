import numpy as np
from abc import ABC, abstractmethod
import graphics

class Object2D:

    @abstractmethod
    def get2DpointList(self):
        pass



class Robot(Object2D):

    total_arm_length = 25.0


    def __init__ (self, number_of_joints = 2, motor_speed = 1.0):
        """[summary]
        
        Keyword Arguments:
            number_of_joints {int} -- must be 1 or more (default: {2})
            motor_speed {float} -- motor_speed = 1 means a rotation of circa 1 degree per step (default: {1.0})
        """        
        self.number_of_joints = number_of_joints
        self.motor_speed = motor_speed
        self.step_size = 0.01745*motor_speed

        self.joints = np.zeros((number_of_joints,), float)
        self.joint_coordinates = np.zeros((number_of_joints, 2), float)

        self.refresh()

  
    def moveArm(self, joint_id, target_rotation):    
        """
        Moves arm in the direction of target_rotation (rotation in radians)
        
        Arguments:
            joint_id {int} -- first joint is 0, second joint 1 etc.
            target_rotation {float} -- rotation in radians
        """        
        normalized_target_rotation = target_rotation%(2*np.pi)

        difference = normalized_target_rotation-self.joints[joint_id]

        if(difference.__abs__<self.step_size):
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
        self.joint_coordinates[0] = [50.0,50.0]

        for i in range (self.number_of_joints-1):
            current_rotation += self.joints[i]
            self.joint_coordinates[i+1][0] = self.joint_coordinates[i][0] + np.sin(current_rotation)*limb_length
            self.joint_coordinates[i+1][1] = self.joint_coordinates[i][1] + np.cos(current_rotation)*limb_length

    def get2DpointList(self):
        return self.joint_coordinates

class Render:

    def __init__ (self, object2DList):
        self.objectList = object2DList
        self.window1 = graphics.GraphWin("window1", 100, 100)


    
    def renderFrame(self):
        
        c = graphics.Point(50,50)
        c.draw(self.window1)
        self.window1.getMouse() # Pause to view result
        # window.close()    # Close window when done

        


myRender = Render("Hi")
print("lets go")
myRender.renderFrame()
        



            


    