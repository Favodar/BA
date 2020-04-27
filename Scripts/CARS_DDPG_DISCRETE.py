import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from CARS_GymEnvironment_DiscreteActions import CustomEnv

my_step_limit = 200
my_step_size = 0.01745
my_number_of_joints = 3

print("CARS_DDPG_DISCRETE.py LESS GO")

env = CustomEnv(step_limit=my_step_limit) # 0.01745*5
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

my_learning_rate = 0.00025 # default: 0.00025
timesteps = 300000
name = "CARS_franka_DISCRETE_DDPG_LR_"  + str(my_learning_rate) + "stepsize_" + str(my_step_size) + "timesteps_" + str(timesteps) + "ep_length_" + str(my_step_limit)
# Configure tensorflow using GPU
# Use tensorboard to show reward over time etc
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))


model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log="/home/fritz/Documents/BA/TensorBoardLogs")
model.learn(total_timesteps= timesteps)



model.save(name)

try:
    f = open("envparameters_" + name, "x")
    f.write(str([my_step_limit, my_step_size, my_number_of_joints]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" + str([my_step_limit, my_step_size, my_number_of_joints]))



while True:
    obs = env.reset()
    obs, rewards, dones, info = env.step([0,0])
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.renderSlow(50)