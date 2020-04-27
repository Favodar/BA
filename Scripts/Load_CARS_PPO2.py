import gym

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from CARS_GymEnvironment_DiscreteActions import CustomEnv
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

# Standard:
# my_signal_rate = 100
# my_signal_repetitions = 15
# my_step_limit = 16

# Shifting Gears:
# my_signal_rate = 100
# my_signal_repetitions = 2
# my_step_limit = 100

filename = "CARS_fixedShape_newObs_ppo2_LR_LinearSchedule_timesteps_1000000ep_length_150turnrate_0.08725maxspeed0.5"

# Load signal parameters from file:
f = open("../Envparameters/envparameters_" + filename, "r")
envparameters = f.read()
envparameters = envparameters.strip('[')
envparameters = envparameters.strip(']')
f_list = [float(i) for i in envparameters.split(",")]
print("envparameters: " + str(f_list))

my_step_limit = int(f_list[0])
my_step_size = float(f_list[1])
my_maxspeed = float(f_list[2])
   

# Initialize environment with signal parameters:
env = CustomEnv(step_limit=my_step_limit, step_size = my_step_size, maxspeed = my_maxspeed) # 0.01745*5

# Load trained model and execute it forever:
model = PPO2.load("../Models/" +filename)

while True:
    #obs = env.reset()
    obs = env.reset()
    #obs = obs.reshape((1,4))
    #print(env.observation_space.shape)
    #obs, rewards, dones, info = env.step([0,0])
    for i in range(1000000): #my_step_limit
        action, _states = model.predict(obs)
        print(action)
        obs, rewards, dones, info = env.step(action)
        #obs = np.array(obs).reshape((1,4))
        env.renderSlow(1000)
    