import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from CARS_GymEnvironment_DiscreteActions import CustomEnv
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

my_step_limit = 150
my_step_size = 0.01745*5
my_maxspeed = 0.5

print("CARS_PPO2_DISCRETE.py LESS GO")

env = CustomEnv(step_limit=my_step_limit, step_size = my_step_size, maxspeed = my_maxspeed) # 0.01745*5
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
timesteps = 1000000
my_learning_rate = LinearSchedule(timesteps, 0.005, 0.0001) # default: 0.00025

name = "CARS_fixedShape_newObs_ppo2_LR_"  + "LinearSchedule_"  + "timesteps_" + str(timesteps) + "ep_length_" + str(my_step_limit) + "turnrate_" + str(my_step_size) + "maxspeed" + str(my_maxspeed)
# Configure tensorflow using GPU
# Use tensorboard to show reward over time etc
model = PPO2(MlpPolicy, env, learning_rate= my_learning_rate.value, verbose=1, tensorboard_log="/home/fritz/Documents/BA/TensorBoardLogs/CARS2") # defaults: learning_rate=2.5e-4,
model.learn(total_timesteps=timesteps, tb_log_name= name)



model.save(name)

try:
    f = open("envparameters_" + name, "x")
    f.write(str([my_step_limit, my_step_size, my_maxspeed]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" + str([my_step_limit, my_step_size, my_maxspeed]))



while True:
    obs = env.reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.renderSlow(50)