import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from FrankaGymEnvironmentEASY import CustomEnv

my_signal_rate = 100
my_signal_repetitions = 15
my_step_limit = 24

env = CustomEnv(signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

my_learning_rate = 0.003 # unused as of now

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

timesteps = 400000
name = "EZBALLS_franka_continuous_ddpg_learning_rate_" + str(my_learning_rate) + "_timesteps_" + str(timesteps)

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log="/home/fritz/Documents/BA/TensorBoardLogs")
model.learn(total_timesteps= timesteps, tb_log_name= name)

model.save(name) # + str(my_learning_rate))

f = open("envparameters_" + name, "x")
f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit,]))
f.close()