#!/usr/bin/env python
"""Implementation of the robot control adapter using ros and gazebo"""
import rospy
import math
# pylint: disable=E0611
from gazebo_msgs.srv import GetPhysicsProperties, GetWorldProperties, \
                            SetPhysicsProperties, AdvanceSimulation
from std_srvs.srv import Empty
import logging

logger = logging.getLogger(__name__)

__author__ = 'NinoCauli'



class RosControlAdapter():
    """
    Represents a robot simulation adapter actually using ROS
    """

    def __init__(self):
        rospy.wait_for_service('/gazebo/get_physics_properties')
        self.__get_physics_properties = rospy.ServiceProxy(
                      'gazebo/get_physics_properties', GetPhysicsProperties, persistent=True)
        rospy.wait_for_service('/gazebo/get_world_properties')
        self.__get_world_properties = rospy.ServiceProxy(
                      'gazebo/get_world_properties', GetWorldProperties, persistent=True)
        rospy.wait_for_service('/gazebo/set_physics_properties')
        self.__set_physics_properties = rospy.ServiceProxy(
                      'gazebo/set_physics_properties', SetPhysicsProperties, persistent=True)
        rospy.wait_for_service('/gazebo/pause_physics')
        self.__pause_client = rospy.ServiceProxy('gazebo/pause_physics', Empty, persistent=True)
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.__unpause_client = rospy.ServiceProxy('gazebo/unpause_physics', Empty, persistent=True)
        rospy.wait_for_service('/gazebo/reset_sim')
        self.__reset = rospy.ServiceProxy('gazebo/reset_sim', Empty, persistent=True)
        rospy.wait_for_service('gazebo/end_world')
        self.__endWorld = rospy.ServiceProxy('gazebo/end_world', Empty, persistent=True)
        rospy.wait_for_service('gazebo/advance_simulation')
        self.__advance_simulation = rospy.ServiceProxy(
                       'gazebo/advance_simulation', AdvanceSimulation, persistent=True)
        self.__time_step = 0.0
        self.__is_initialized = False


    def initialize(self):
        """
        Initializes the world simulation control adapter
        """
        if not self.__is_initialized:
            physics = self.__get_physics_properties()
            paused = physics.pause
            if not paused:
                self.__pause_client()
            self.__reset()
            self.__time_step = physics.time_step
            self.__is_initialized = True

        logger.info("Robot control adapter initialized")
        return self.__is_initialized

    @property

    def time_step(self):
        """
        Gets the physics simulation time step in seconds

        :param dt: The physics simulation time step in seconds
        :return: The physics simulation time step in seconds
        """
        return self.__time_step


    def set_time_step(self, time_step):
        """
        Sets the physics simulation time step in seconds

        :param time_step: The physics simulation time step in seconds
        :return: True, if the physics simulation time step is updated, otherwise False
        """
        physics = self.__get_physics_properties()
        success = self.__set_physics_properties(
            time_step,
            physics.max_update_rate,
            physics.gravity,
            physics.ode_config)
        if success:
            self.__time_step = time_step
            logger.info("new time step = %f", self.__time_step)
        else:
            logger.warn("impossible to set the new time step")
        return success

    @property

    def is_paused(self):
        """
        Queries the current status of the physics simulation

        :return: True, if the physics simulation is paused, otherwise False
        """
        physics = self.__get_physics_properties()
        paused = physics.pause
        return paused

    @property

    def is_alive(self):
        """
        Queries the current status of the world simulation

        :return: True, if the world simulation is alive, otherwise False
        """
        logger.debug("Getting the world properties to check if we are alive")
        world = self.__get_world_properties()
        success = world.success
        return success


    def run_step(self, dt):
        """
        Runs the world simulation for the given CLE time step in seconds

        :param dt: The CLE time step in seconds
        """
        if math.fmod(dt, self.__time_step) < 1e-10:
            steps = dt / self.__time_step
            logger.debug("Advancing simulation")
            self.__advance_simulation(steps)
        else:
            logger.error("dt is not multiple of the physics time step")
            raise ValueError("dt is not multiple of the physics time step")


    def shutdown(self):
        """
        Shuts down the world simulation
        """
        logger.info("Shutting down the world simulation")
        # Do not call endWorld here, it makes Gazebo Stop !


    def reset(self):
        """
        Resets the physics simulation
        """
        logger.info("Resetting the world simulation")
        self.__reset()


    def unpause(self):
        """
        Unpaused the physics
        """
        logger.info("Unpausing the world simulation")
        if self.is_paused:
            self.__unpause_client()


    def pause(self):
        """
        Pause the physics
        """
        logger.info("Pausing the world simulation")
        if not self.is_paused:
            self.__pause_client()

