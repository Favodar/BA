import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule
from NEW_FrankaGymEnvironment_DiscreteActions import CustomEnv

my_signal_rate = 100
my_signal_repetitions = 25
my_step_limit = 12

env = CustomEnv(signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

timesteps = 100000
scheduler = LinearSchedule(timesteps, 0.001, 0.0001)
my_learning_rate = scheduler.value # 0.0005 

model = PPO2(MlpPolicy, env, learning_rate= my_learning_rate, verbose=1, tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/NEW_FRANKA") # defaults: learning_rate=2.5e-4,
model.learn(total_timesteps= timesteps)

name = "NEW_ppo2_franka_discrete_ppo2"  + str(my_learning_rate) + "_timesteps_" + str(timesteps)
model.save(name)

f = open("envparameters_" + name, "x")
f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit]))
f.close()