import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule
from NEW_FrankaGymEnvironment_DiscreteActions import CustomEnv

filename = "Continued_CD6_LoadedFrom250kSteps_Phys002_ppo2_franka_discrete_LR_0.005-0.0001_timesteps_100000srate_sreps_slimit_1002512"
model_iteration = "_1"
#filename = "NEW_CRAZYDEEP5_ppo2_franka_discrete_LR_0.001-0.0001_timesteps_10000srate_sreps_slimit_1002512"
# Load signal parameters from file:
f = open("../Envparameters/envparameters_" + filename, "r")
envparameters = f.read()
envparameters = envparameters.strip('[')
envparameters = envparameters.strip(']')
f_list = [int(i) for i in envparameters.split(",")]
print(str(f_list))

my_signal_rate = f_list[0]
my_signal_repetitions = f_list[1]
my_step_limit = f_list[2]

# CAREFUL, HARDCODED STEPLIMIT:
my_step_limit = 18
my_physics_stepsize = 0.002

# Initialize environment with signal parameters:
env = CustomEnv(signal_rate=my_signal_rate,
                signal_repetitions=my_signal_repetitions, step_limit=my_step_limit, physics_stepsize= my_physics_stepsize)

env = DummyVecEnv([lambda: env])

#filename = "NEW_CRAZYDEEP6_GPU_ppo2_franka_discrete_LR_0.001-0.0001_timesteps_100000srate_sreps_slimit_1002512"
#filename = "NEW_CRAZYDEEP6_GPU_ppo2_franka_discrete_LR_0.001-0.0001_timesteps_100000srate_sreps_slimit_1002512"

model = PPO2.load("../../Models/" + filename + model_iteration, env=env) # tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/NEW_DEEP_FRANKA"
model.tensorboard_log = "/media/ryuga/Shared Storage/TensorBoardLogs/NEW_DEEP_FRANKA2"
# env = DummyVecEnv([lambda: env])
# model.set_env(env)
# print("TB Log: ")+ str(model.tensorboard_log)



for i in range(1):
    obs = env.reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if(dones):
            break

print("NEW_DEEP_LOADED_PPO2_Franka.py LESS GO")

timesteps = 11200000
lr_start = 0.0005
lr_end = 0.0001
scheduler = LinearSchedule(schedule_timesteps= timesteps,initial_p= lr_start, final_p = lr_end)
# my_learning_rate = scheduler.value
my_learning_rate = 0.00025  # scheduler.value default: 2.5e-4=0.00025
#print_LR = str(lr_start) + "-" + str(lr_end)
print_LR = str(my_learning_rate)

model.learning_rate = my_learning_rate

# name = filename
name = "CD6_1.5xSlimit_staticLR_LoadedFrom270kSteps_Phys004_ppo2_franka_discrete_LR_" + print_LR + "_timesteps_" + \
    str(timesteps) + "srate_sreps_slimit_" + str(my_signal_rate) + \
    str(my_signal_repetitions) + str(my_step_limit)

try:
    f = open("../Envparameters/envparameters_" + name, "x")
    f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" +
          str([my_signal_rate, my_signal_repetitions, my_step_limit]))


i = 0
while(i <= (timesteps/10000)):
    model.learn(total_timesteps=10000, tb_log_name=name, log_interval=10, reset_num_timesteps=False)
    model.save("../Models/" + name + "_" + str(i))
    i+=1

# model.learn(total_timesteps=timesteps, tb_log_name=name, reset_num_timesteps=False, log_interval=10)
# model.save("../Models/" + name + "_" + str(i))
