import gym

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from CARS_TRIAL_GymEnvironment_DiscreteActions import CustomEnv
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

my_step_limit = 120
my_step_size = 0.01745*11.25
my_maxspeed = 2.5
my_acceleration = 2.5/4
my_randomBall = True
my_binaryReward = True

print("CARS_PPO2_DISCRETE.py LESS GO")

#env = CustomEnv(step_limit=my_step_limit, step_size = my_step_size, maxspeed = my_maxspeed,acceleration=my_acceleration, randomBall = my_randomBall, binaryReward= my_binaryReward) # 0.01745*5
env = make_vec_env(CustomEnv, n_envs=16, env_kwargs={'step_limit':my_step_limit, 'step_size' : my_step_size, 'maxspeed' : my_maxspeed, 'acceleration' : my_acceleration, 'randomBall' : my_randomBall, 'binaryReward' : my_binaryReward})
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
timesteps = 2000000
my_learning_rate = LinearSchedule(timesteps, 0.005, 0.0001) # default: 0.00025

name = "CARS_medium5_225_LSTM_ppo2_LR_"  + "LinearSchedule_"  + "timesteps_" + str(timesteps) + "ep_length_" + str(my_step_limit) + "turnrate_" + str(my_step_size) + "maxspeed_" + str(my_maxspeed) + "randomBall_" + str(my_randomBall) + "binaryReward_" + str(my_binaryReward)
# Configure tensorflow using GPU
# Use tensorboard to show reward over time etc
model = PPO2(MlpLstmPolicy, env,nminibatches=16, learning_rate= my_learning_rate.value, verbose=1, tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/CARSTRIAL") # defaults: learning_rate=2.5e-4,
model.learn(total_timesteps=timesteps, tb_log_name= name)



model.save("../Models/" + name)

try:
    f = open("../Envparameters/envparameters_" + name, "x")
    f.write(str([my_step_limit, my_step_size, my_maxspeed, my_acceleration, my_randomBall, my_binaryReward]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" + str([my_step_limit, my_step_size, my_maxspeed, my_acceleration, my_randomBall, my_binaryReward]))

env = DummyVecEnv([lambda: env])

while True:
    obs = env.envs[0].reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.envs[0].step(action)
        env.envs[0].renderSlow(25)