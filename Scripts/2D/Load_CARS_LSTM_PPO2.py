import gym

import numpy as np

from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from CARS_TRIAL_GymEnvironment_DiscreteActions import CustomEnv
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

filename = "CARS_medium5_225_LSTM_ppo2_LR_LinearSchedule_timesteps_2000000ep_length_120turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True"
#filename = "CARS_medium5_225_newObs_ppo2_LR_LinearSchedule_timesteps_4000000ep_length_120turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True"
# Load signal parameters from file:
f = open("../Envparameters/envparameters_" + filename, "r")
envparameters = f.read()
envparameters = envparameters.strip('[')
envparameters = envparameters.strip(']')
f_list = [i for i in envparameters.split(",")]
print("envparameters: " + str(f_list))

my_step_limit = int(f_list[0])
my_step_size = float(f_list[1])
my_maxspeed = float(f_list[2])
my_acceleration = float(f_list[3])
my_randomBall = bool(f_list[4])
my_binaryReward = bool(f_list[5])

# Initialize environment with signal parameters:
env = make_vec_env(CustomEnv, n_envs=16, env_kwargs={'step_limit':my_step_limit, 'step_size' : my_step_size, 'maxspeed' : my_maxspeed, 'acceleration' : my_acceleration, 'randomBall' : my_randomBall, 'binaryReward' : my_binaryReward})

# Load trained model and execute it forever:
model = PPO2.load("../Models/" +filename)

while True:
    obs = env.reset()
    for i in range(my_step_limit*2): 
        action, _states = model.predict(obs)
        #print(obs)
        obs, rewards, dones, info = env.step(action)
        print(rewards)
        for environment in env.envs:
            environment.renderSlow(50)
        # if(dones):
        #      env.render()
        #      break
        #env.envs[0].renderSlow(50)

# while True:
#     obs = env.reset()
#     #obs = env.envs[0].reset()
#     #obs = obs.reshape((1,4))
#     #print(env.observation_space.shape)
#     #obs, rewards, dones, info = env.step([0,0])
#     for i in range(my_step_limit*2): #my_step_limit
#         action, _states = model.predict(obs)
#         #print(action)
#         print(obs[0])
#         obs, rewards, dones, info = env.step(action)
#         #obs, rewards, dones, info = env.envs[0].step(action)
#         #obs = np.array(obs).reshape((1,4))
#         if(dones[0]):
#              env.envs[0].renderSlow(1)
#              break
#         env.envs[0].renderSlow(10)
        
    