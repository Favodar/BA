import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule
from NEW_Efficient_FrankaGymEnvironment_DiscreteActions import CustomEnv

""" History

filename = "i7_CD7SL_Monday_SaveIntervall500_LogLR_Phys008_ppo2_franka_discrete_LR_0.001-0_timesteps_4000000srate_sreps_slimit_1002550"
model_iteration = "_639"
pretraining_steps_with_new_LR = 250000
name = "RYZEN_CD7SL_MondayContinued_FromCa250k_SI500_LogLR_Phys008_ppo2_franka_discrete_LR_"

filename = "RYZEN_CD7SL_GrillenNightlyContinued2_SaveIntervall500_LogLR_Phys008_ppo2_franka_discrete_LR_0.001-0.0_timesteps_1000000srate_sreps_slimit_1002550"
model_iteration = "_759"
pretraining_steps_with_new_LR = 300000 + 280000

filename = "Ryzen4400_CD7_EfficientCode_Phys008_constLR_ppo2_franka_discrete_LR_0.004_timesteps_4000000srate_sreps_slimit_1002550"
model_iteration = "_14"
pretraining_steps_with_new_LR = 150000
name = "ryzen4300_thursday_carsLR_EPlength12sec_From150k_CD7_Phys008_ppo2_franka_discrete_LR"

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

filename = "Ryzen_DefNN_EASY_ELR_Phys006_ppo2_franka_discrete_LR_0.001-6.3e-05_DefaultNN_timesteps_1000000srate_sreps_slimit_1002550joints_2"
model_iteration = "_15"
pretraining_steps_with_new_LR = 320000
#pretraining_steps_with_new_LR = 1240000 + 580000
#filename = "NEW_CRAZYDEEP5_ppo2_franka_discrete_LR_0.001-0.0001_timesteps_10000srate_sreps_slimit_1002512"
# Load signal parameters from file:
f = open("../Envparameters/envparameters_" + filename, "r")
envparameters = f.read()
envparameters = envparameters.strip('[')
envparameters = envparameters.strip(']')
f_list = [float(i) for i in envparameters.split(",")]
print("envparameters loaded from file [my_signal_rate, my_signal_repetitions, my_step_limit, lr_start, lr_end, timesteps]:")
print(str(f_list))

my_signal_rate = int(f_list[0])
my_signal_repetitions = int(f_list[1])
my_step_limit = int(f_list[2])
my_lr_start = f_list[3]
my_lr_end = f_list[4]
my_timesteps = int(f_list[5])
try:
    my_number_of_joints = int(f_list[6])
except:
    print("number of joints couldnt be loaded and is defaulted to 7")
    my_number_of_joints = 7


# my_step_limit = 50
my_physics_stepsize = 0.004

# Initialize environment with signal parameters:
env = CustomEnv(signal_rate=my_signal_rate,
                signal_repetitions=my_signal_repetitions, step_limit=my_step_limit, physics_stepsize= my_physics_stepsize, number_of_joints=my_number_of_joints)

env = DummyVecEnv([lambda: env])

#filename = "NEW_CRAZYDEEP6_GPU_ppo2_franka_discrete_LR_0.001-0.0001_timesteps_100000srate_sreps_slimit_1002512"
#filename = "NEW_CRAZYDEEP6_GPU_ppo2_franka_discrete_LR_0.001-0.0001_timesteps_100000srate_sreps_slimit_1002512"

#model = PPO2.load("../../Models/" + filename + model_iteration, env=env) # tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/NEW_DEEP_FRANKA"
# tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/NEW_DEEP_FRANKA"
model = PPO2.load("/media/ryuga/TOSHIBA EXT/BA/Models/" +
                  filename + model_iteration, env=env)
model.tensorboard_log = "/media/ryuga/TOSHIBA EXT/BA/TensorBoardLogs/NEW_DEEP_FRANKA_EASY_RYZEN"
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

print("NEW_DEEP_LOG_LOADED_PPO2_Franka.py LESS GO")

# Cars Trial Timesteps & Learning Rate (starting slightly lower):
timesteps = my_timesteps #4000000

lr_start = my_lr_start #0.004
lr_end = my_lr_end #0.0001
# scheduler = LinearSchedule(timesteps, lr_start, lr_end)
# my_learning_rate = scheduler.value
my_learning_rate = 0.000063  # scheduler.value default: 2.5e-4=0.00025
#print_LR = str(lr_start) + "-" + str(lr_end)
print_LR = str(my_learning_rate)

is_static_lr = True
model.learning_rate = 0.000063 #lr_start

# name = filename
name = "Ryzen_cFrom320kWithStaticLR_DefNN_EASY_ELR_Phys006_ppo2_franka_discrete_LR_" + print_LR + "_timesteps_" + \
    str(timesteps) + "srate_sreps_slimit_" + str(my_signal_rate) + \
    str(my_signal_repetitions) + str(my_step_limit)

try:
    f = open("../Envparameters/envparameters_" + name, "x")
    f.write(str([my_signal_rate, my_signal_repetitions,
                 my_step_limit, lr_start, lr_end, timesteps]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" +
          str([my_signal_rate, my_signal_repetitions, my_step_limit, lr_start, lr_end, timesteps]))

#pre_training_save_interval = 20000
#pretraining_iterations = pretraining_steps_with_new_LR/pre_training_save_interval

lr_update_interval = 500
save_interval = 20000
modulo_number = save_interval/lr_update_interval

#lr_stepsize = (lr_start-lr_end)/(timesteps/save_interval)
print("lr_start: " + str(lr_start))
print("log formula: " + str(lr_start*0.5 **
                            (((i+(pretraining_steps_with_new_LR/lr_update_interval))*lr_update_interval)*(10/timesteps))))
i = 0
if(is_static_lr):
    while(True):  # i <= (timesteps/save_interval)):
        model.learn(total_timesteps=save_interval, tb_log_name=name,
                    log_interval=10, reset_num_timesteps=False)
        model.save("/media/ryuga/TOSHIBA EXT/BA/Models/" + name + "_" + str(i))
        #dyn_lr.count()
        i += 1

else:
    while(i <= (timesteps/lr_update_interval)):
        # linear: model.learning_rate = lr_start-(lr_stepsize*(i+pretraining_iterations))
        # log:
        # model.learning_rate = lr_start*0.5**(((i*lr_update_interval)+pretraining_steps_with_new_LR)*(10/timesteps))
        # static: model.learning_rate = static_learning_rate
        model.learn(total_timesteps=lr_update_interval, tb_log_name=name,
                    log_interval=10, reset_num_timesteps=False)
        if(i % modulo_number == (modulo_number-1)):
            model.save("/media/ryuga/TOSHIBA EXT/BA/Models/" + name + "_" + str(i/modulo_number))
        i += 1
