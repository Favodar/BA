# BA
An environment for training a virtual robot with deep reinforcement learning (deep RL). It uses the deep RL library stable-baselines by OpenAI, the Robot Operating System (ROS), and the robot simulator Gazebo 9 with the physics engine ODE. I use it with a virtual model of the Franka Emika Panda cobot. If you want to use this project or have any questions, feel free to ask. Unfortunately the setup is complicated, and I wouldn't recommend trying it on your own if you have no experience with ROS, but I may be able to help you.

The general architecture is this:
On the surface is a script that controls the hyperparameters, starts & ends the training and saves the logs and models (BallrollingTouchEnd_PPO2_Franka.py)  
This script imports an environment (in the RL sense) which implements the OpenAI Gym environment class (BallrollingTouchEnd_FrankaGymEnvironment_ContinuousActions.py).  
This environment has high-level control over the robot. The lower-level control (which isn't really low-level) and the reward calculation are implemented in the GymReward class (BallrollingTouchEnd_FrankaRewardNode_Efficient.py).  

Those are the classes I have written and uploaded here. They import other existing libraries and classes, which are connected as follows:
The lower-level control class imports rospy (python library for ROS), and therefore connects everything with ROS, which in turn connects to the physics & robot simulation Gazebo. This means, the impact of the control signals on the robot can eventually be observed via the Gazebo Client (which is the Gazebo GUI).
The surface-level script imports OpenAI's stable-baselines library, in order to use its implementations of deep reinforcement learning (deep RL) algorithms like PPO2 and DDPG.

This reinforcement learning project was done for my Bachelor thesis. What follows is the introduction of the Thesis:

### Introduction
## Motivation
Collaborative robots, or cobots, are a relatively new phenomenon with the potential to transform the workplace in many industries. The hope is that they will be able to execute menial tasks, freeing up human workers for more complex, strategic and interesting work, being a welcomed relief and not a competitor. My goal in this project is to give a cobot the ability to learn simple tasks on its own, opening the door for the acquisition of more complex tasks in the future.
For this work, I use the cobot Panda by German manufacturer Franka Emika, and aim for the mastery of a task that comes natural to humans, but poses a sufficiently complex challenge to a robot: throwing a ball at, or rolling a ball towards, different targets with randomly assigned positions, by making an appropriate arm movement and releasing the ball from its grasp at the right time.
Making the agent learn simpler tasks like touching a specified target or touching a randomly selected target should serve as stepping stones.

Importantly, the cobot shall not be shown how to throw the ball, nor shall it be given any prior knowledge about ballistic trajectories or physical laws and equations in general.
Instead, the objective for the robot is to learn everything from scratch by trial and error, only being given a goal, but no instructions on how to achieve that goal. These requirements are a perfect fit for a machine learning technique called reinforcement learning (RL), which uses reward and punishment to teach an agent abilities by trial and error.
This way, no domain specific expert knowledge is required to teach the robot the skill. Additionally, with this approach, the Panda can learn things where it is impossible to specify every single action for every situation, i.e. tasks with nearly infinite, context dependent solutions.
2.2 Implementation
It would not be feasible to execute the training in the real world with the real robot, because in reinforcement learning (RL), usually at least several thousand training episodes are required. In complex tasks, this number can be much higher. For example, the OpenAI Five network completes 180 years worth of training every day (OpenAI 2018).
In addition, with a real life robot, it would be necessary to automate the processes of moving the target, and moving the ball back into the hands of the Panda. Even if this automation would be successfully implemented and the training could be made efficient enough to complete in weeks, this would render the robot unusable for other tasks for weeks, even in the unlikely case of getting the training setup and parameters right on the first try.
A much more feasible solution, although not without challenges, is simulating a virtual robot and environment. This makes generating new targets and resetting the throwing object almost trivial. Most importantly, it allows for training in faster-than-real-time, while not actually using the real cobot or other unique and expensive equipment.

For everything to come together, an accurate virtual model of the Franka Emika Panda needs to be connected to an RL algorithm in a manner that is transferable to the real world, i.e. the communication needs happen in the same way in which it takes place with the real cobot. This can be achieved by using a robotics middleware.
2.3 Contribution
With the creation of a functioning framework for training a virtual version of the Panda with deep reinforcement learning, a foundation is laid for applying RL to robots and especially the Panda in the Innovationshub, a think tank associated with the University of Applied Sciences Düsseldorf.
