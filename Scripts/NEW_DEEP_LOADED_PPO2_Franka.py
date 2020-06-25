import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule
from NEW_Efficient_FrankaGymEnvironment_DiscreteActions import CustomEnv

""" History
filename = "ryzen4400_heimbach_carsRepairedLR2_EPlength12sec_From760k_CD6_Phys004_ppo2_franka_discrete_LR0.004-0.0001_timesteps_4000000srate_sreps_slimit_1002550"
model_iteration = "_65"
pretraining_steps_with_new_LR = 720000

filename = "cFrom50k_CD5_Phys002_constLR_ppo2_franka_discrete_LR0.00075_timesteps_11200000srate_sreps_slimit_1002518"
model_iteration = "_6"
name = "cFrom120k_nightly_CD5_Phys002_constLR_ppo2_franka_discrete_LR"

filename = "Continued_CD6_LoadedFrom250kSteps_Phys002_ppo2_franka_discrete_LR_0.005-0.0001_timesteps_100000srate_sreps_slimit_1002512"
model_iteration = "_1"
name = "CD6_1.5xSlimit_staticLR_LoadedFrom270kSteps_Phys004_ppo2_franka_discrete_LR_"
"""

filename = "Ryzen4400_CD7_EfficientCode_Phys008_constLR_ppo2_franka_discrete_LR_0.004_timesteps_4000000srate_sreps_slimit_1002550"
model_iteration = "_14"
pretraining_steps_with_new_LR = 150000
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
my_step_limit = 50
my_physics_stepsize = 0.004

# Initialize environment with signal parameters:
env = CustomEnv(signal_rate=my_signal_rate,
                signal_repetitions=my_signal_repetitions, step_limit=my_step_limit, physics_stepsize= my_physics_stepsize)

env = DummyVecEnv([lambda: env])

#filename = "NEW_CRAZYDEEP6_GPU_ppo2_franka_discrete_LR_0.001-0.0001_timesteps_100000srate_sreps_slimit_1002512"
#filename = "NEW_CRAZYDEEP6_GPU_ppo2_franka_discrete_LR_0.001-0.0001_timesteps_100000srate_sreps_slimit_1002512"

#model = PPO2.load("../../Models/" + filename + model_iteration, env=env) # tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/NEW_DEEP_FRANKA"
# tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/NEW_DEEP_FRANKA"
model = PPO2.load("/media/ryuga/Shared Storage/Models/" +
                  filename + model_iteration, env=env)
model.tensorboard_log = "/media/ryuga/Shared Storage/TensorBoardLogs/NEW_DEEP_FRANKA4"
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

# Cars Trial Timesteps & Learning Rate (starting slightly lower):
timesteps = 4000000

lr_start = 0.004
lr_end = 0.0001
# scheduler = LinearSchedule(timesteps, lr_start, lr_end)
# my_learning_rate = scheduler.value
# my_learning_rate = 0.00075  # scheduler.value default: 2.5e-4=0.00025
print_LR = str(lr_start) + "-" + str(lr_end)
# print_LR = str(my_learning_rate)

model.learning_rate = lr_start

# name = filename
name = "ryzen4300_thursday_carsLR_EPlength12sec_From150k_CD7_Phys008_ppo2_franka_discrete_LR" + print_LR + "_timesteps_" + \
    str(timesteps) + "srate_sreps_slimit_" + str(my_signal_rate) + \
    str(my_signal_repetitions) + str(my_step_limit)

try:
    f = open("../Envparameters/envparameters_" + name, "x")
    f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" +
          str([my_signal_rate, my_signal_repetitions, my_step_limit]))

save_interval = 20000
pre_training_save_interval = 10000
pretraining_iterations = pretraining_steps_with_new_LR/pre_training_save_interval

lr_stepsize = (lr_start-lr_end)/(timesteps/save_interval)
print("lr_stepsize: " + str(lr_stepsize))

i = 0
while(i <= (timesteps/save_interval)):
    model.learning_rate = lr_start-(lr_stepsize*(i+pretraining_iterations))
    model.learn(total_timesteps=save_interval, tb_log_name=name, log_interval=10, reset_num_timesteps=False)
    model.save("/media/ryuga/Shared Storage/Models/" + name + "_" + str(i))
    i+=1

# model.learn(total_timesteps=timesteps, tb_log_name=name, reset_num_timesteps=False, log_interval=10)
# model.save("../Models/" + name + "_" + str(i))
