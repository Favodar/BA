import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from FrankaGymEnvironment_DiscreteActions import CustomEnv

my_signal_rate = 100
my_signal_repetitions = 2
my_step_limit = 100

env = CustomEnv(signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

my_learning_rate = 0.005

model = PPO2(MlpPolicy, env, learning_rate= my_learning_rate, verbose=1) # defaults: learning_rate=2.5e-4,
model.learn(total_timesteps=10000)

name = "ppo2_franka_discrete"
model.save(name)

f = open("envparameters_" + name, "x")
f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit]))
f.close()