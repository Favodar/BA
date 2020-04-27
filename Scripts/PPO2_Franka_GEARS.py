import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from FrankaGymEnvironment_ShiftingGears import CustomEnv

my_signal_rate = 100
my_signal_repetitions = 15
my_step_limit = 24
my_number_of_gears = 2
my_gear_interval = 0.1
sps = my_signal_rate/my_signal_repetitions # signals per second

env = CustomEnv(signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit, number_of_gears= my_number_of_gears, gear_interval= my_gear_interval)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

my_learning_rate = 0.003 # 0.01 is probably a good value for training <1h
timesteps = 40000
model = PPO2(MlpPolicy, env, learning_rate= my_learning_rate, verbose=1, tensorboard_log="/home/fritz/Documents/BA/TensorBoardLogs") # defaults: learning_rate=2.5e-4,
model.learn(total_timesteps= timesteps)

name = "BALLS_ppo2_franka_SHIFTING_GEARS_learning_rate_" + str(my_learning_rate) + "_sps_" + str(sps) + "_timesteps_" + str(timesteps)
model.save(name) # + str(my_learning_rate))

f = open("envparameters_" + name, "x")
f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit, my_number_of_gears, my_gear_interval]))
f.close()