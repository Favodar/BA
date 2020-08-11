import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from Franka2DGymEnvironment_DiscreteActions import CustomEnv
from My_Dynamic_Learning_Rate import ExpLearningRate

my_step_limit = 200
my_step_size = 0.01745*5
my_number_of_joints = 3

print("TWOD_PPO2_Franka_DISCRETE.py LESS GO")

env = CustomEnv(step_limit=my_step_limit, step_size = my_step_size, number_of_joints = my_number_of_joints) # 0.01745*5
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

my_learning_rate = 0.00025 # default: 0.00025
timesteps = 800000
save_interval = 0
is_static_lr = False

# for dynamic LRs:
lr_start = 0.0001
lr_end = 0.00005#0.000063
half_life = 1.0
dyn_lr = ExpLearningRate(
    timesteps=timesteps, lr_start=lr_start, lr_min=lr_end, half_life=half_life, save_interval=save_interval)
my_learning_rate = dyn_lr.value  # 0.0005
print_LR = str(lr_start) + "-" + str(lr_end)

name = "TWOD_RANDOMBALL_RepairedRobot_400k2_franka_DISCRETE_ppo2_LR_"  + print_LR + "stepsize_" + str(my_step_size) + "timesteps_" + str(timesteps) + "ep_length_" + str(my_step_limit)
# Configure tensorflow using GPU
# Use tensorboard to show reward over time etc
model = PPO2(MlpPolicy, env, learning_rate=my_learning_rate, verbose=1,
             tensorboard_log="/media/ryuga/TOSHIBA EXT/BA/TensorBoardLogs/TWOD_RANDOM_revisited")  # defaults: learning_rate=2.5e-4,
model.learn(total_timesteps=timesteps, tb_log_name= name)


model.save("/media/ryuga/TOSHIBA EXT/BA/Models/" + name)

f = open("envparameters_" + name, "x")
f.write(str([my_step_limit, my_step_size, my_number_of_joints]))
f.close()


while True:
    obs = env.reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
