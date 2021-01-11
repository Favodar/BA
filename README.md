# BA
An environment for training a virtual robot with deep reinforcement learning (deep RL). It uses the deep RL library stable-baselines by OpenAI, the Robot Operating System (ROS), and the robot simulator Gazebo 9 with the physics engine ODE. I use it with a virtual model of the Franka Emika Panda cobot. If you want to use this project or have any questions, feel free to ask. Unfortunately the setup is complicated, and I wouldn't recommend trying it on your own if you have no experience with ROS, but I may be able to help you.

The general architecture is this:
On the surface is a script that controls the hyperparameters, starts & ends the training and saves the logs and models (BallrollingTouchEnd_PPO2_Franka.py)  
This script imports an environment (in the RL sense) which implements the OpenAI Gym environment class (BallrollingTouchEnd_FrankaGymEnvironment_ContinuousActions.py).  
This environment has high-level control over the robot. The lower-level control (which isn't really low-level) and the reward calculation are implemented in the GymReward class (BallrollingTouchEnd_FrankaRewardNode_Efficient.py).  

Those are the classes I have written and uploaded here. They import other existing libraries and classes, which are connected as follows:
The lower-level control class imports rospy (python library for ROS), and therefore connects everything with ROS, which in turn connects to the physics & robot simulation Gazebo. This means, the impact of the control signals on the robot can eventually be observed via the Gazebo Client (which is the Gazebo GUI).
The surface-level script imports OpenAI's stable-baselines library, in order to use its implementations of deep reinforcement learning (deep RL) algorithms like PPO2 and DDPG.
