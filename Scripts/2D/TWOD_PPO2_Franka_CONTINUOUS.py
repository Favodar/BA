import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from Franka2DGymEnvironmentNOREPS import CustomEnv

my_step_limit = 400
my_step_size = 0.01745*5
my_number_of_joints = 3

print("TWOD_PPO2_Franka_CONTINUOUS.py LESS GO")

env = CustomEnv(step_limit=my_step_limit, step_size = my_step_size, number_of_joints = my_number_of_joints) # 0.01745*5
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

my_learning_rate = 0.00025 # default: 0.00025
timesteps = 100000000
name = "TWOD_OLDVERSION_FIXEDBALL_franka_CONTINUOUS_ppo2_LR_"  + str(my_learning_rate) + "stepsize_" + str(my_step_size) + "timesteps_" + str(timesteps) + "ep_length_" + str(my_step_limit)
# Configure tensorflow using GPU
# Use tensorboard to show reward over time etc
model = PPO2(MlpPolicy, env, learning_rate= my_learning_rate, verbose=1, tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/TWOD_NEW") # defaults: learning_rate=2.5e-4,
model.learn(total_timesteps=timesteps, tb_log_name= name)


model.save(name)

try:
    f = open("envparameters_" + name, "x")
    f.write(str([my_step_limit, my_step_size, my_number_of_joints]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" + str([my_step_limit, my_step_size, my_number_of_joints]))


while True:
    obs = env.reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()