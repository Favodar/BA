import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from Franka2DGymEnvironmentNOREPS import CustomEnv

my_signal_repetitions = 15
my_step_limit = 100

env = CustomEnv()
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

my_learning_rate = 0.003 # unused as of now

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

timesteps = 100000
name = "TWODNOREPS_franka_continuous_ddpg_learning_rate_" + "_timesteps_" + str(timesteps)

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log="/home/fritz/Documents/BA/TensorBoardLogs")
model.learn(total_timesteps= timesteps, tb_log_name= name)

model.save(name) # + str(my_learning_rate))

# f = open("envparameters_" + name, "x")
# f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit,]))
# f.close()

while True:
    obs = env.reset()
    for i in range(500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()