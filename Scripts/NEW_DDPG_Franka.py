from My_Dynamic_Learning_Rate import ExpLearningRate
import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from NEW_Efficient_FrankaGymEnvironment_DiscreteActions import CustomEnv




my_signal_rate = 100
my_signal_repetitions = 25
my_step_limit = 50
my_number_of_joints = 2

env = CustomEnv(signal_rate=my_signal_rate, signal_repetitions=my_signal_repetitions,
                step_limit=my_step_limit, number_of_joints=my_number_of_joints)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
architecture_name = "UnknownArchitecture"

timesteps = 1000000
save_interval = 20000
is_static_lr = False
lr_start = 0.001
lr_end = 0.000063
half_life = 0.1
dyn_lr = ExpLearningRate(
    timesteps=timesteps, lr_start=lr_start, lr_min=lr_end, half_life=half_life, save_interval=save_interval)

#scheduler = LinearSchedule(schedule_timesteps= timesteps,initial_p= lr_start, final_p = lr_end)
# 0.000063  # scheduler.value # 0.0005 default: 2.5e-4=0.00025
my_learning_rate = dyn_lr.value
#print_LR = str(my_learning_rate)
print_LR = str(lr_start) + "-" + str(lr_end)

p_quarks = None
# run PPO2_2 p_quarks = dict(net_arch=[128, dict(vf=[256, 256])])
# run PPO2_3 p_quarks = dict(net_arch=[128, 128, dict(vf=[256, 256, 256], pi=[16 ,16 , 16])])
# run CRAZYDEEP2 p_quarks = dict(net_arch=[128, 128, 128, dict(vf=[256, 256, 256, 256, 256], pi=[32, 16, 32])])
# run CRAZDYDEEP3 p_quarks = dict(net_arch=[128, 128, 128, dict(
#    vf=[1024, 256, 256, 256, 256, 16], pi=[32, 32, 16])])
# run CRAZYDEEP4 p_quarks = dict(net_arch=[1024, 1024, 1024, dict(
#        vf=[512, 512, 512, 512, 512, 256], pi=[64, 64, 32])])

#CRAZYDEEP5:
# p_quarks = dict(net_arch=[8192, 4096, 4096, dict(
#     vf=[2048, 2048, 2048, 2048, 2048, 1024], pi=[64, 64, 32])])

# CRAZYDEEP6 p_quarks = dict(net_arch=[dict(
#    vf=[1024, 1024, 1024, 1024, 1024], pi=[256, 256, 128])])

#CRAZYDEEP7:
#p_quarks = dict(net_arch=[8192, 8192, dict(
#    vf=[8192, 4096, 4096, 2048], pi=[256, 256, 128])])

#CRAZYDEEP7 Lite:
#p_quarks = dict(net_arch=[4096, 4096, dict(
#    vf=[4096, 2048, 2048, 1024], pi=[256, 256, 128])])

#CRAZYDEEP7 SuperLite:
#p_quarks = dict(net_arch=[1024, 1024, dict(
#    vf=[1024, 512, 512, 256], pi=[256, 256, 128])])

#architecture_name = "ReasonableDeep1"
#p_quarks = dict(net_arch=[dict(
#    vf=[128, 128, 128], pi=[128, 128, 128])])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

if(p_quarks == None):
    model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise,
                 tensorboard_log="/media/ryuga/TOSHIBA EXT/BA/TensorBoardLogs/NEW_DEEP_FRANKA5_RYZEN")  # defaults: learning_rate=2.5e-4,
    print("Warning: default network architecture")
    architecture_name = "DefaultNN"
else:
    model = DDPG(MlpPolicy, env, verbose=1, policy_kwargs= p_quarks, param_noise=param_noise, action_noise=action_noise,
                 tensorboard_log="/media/ryuga/TOSHIBA EXT/BA/TensorBoardLogs/NEW_DEEP_FRANKA5_RYZEN")  # defaults: learning_rate=2.5e-4,
    print("Neural net architecture: " + architecture_name)





name = "RYZEN_DDPG_2joints_withNoise_NewEffCode_DefaultNN_LogLR_Phys004_ddpg_franka_continuous_LR_" + print_LR + "_timesteps_" + \
    str(timesteps) + "srate_sreps_slimit_" + str(my_signal_rate) + \
    str(my_signal_repetitions) + str(my_step_limit)



try:
    f = open("../Envparameters/envparameters_" + name, "x")
    f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit,
                 lr_start, lr_end, timesteps, my_number_of_joints]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" +
          str([my_signal_rate, my_signal_repetitions, my_step_limit, lr_start, lr_end, timesteps, my_number_of_joints]))


i = 0

while(True):  # i <= (timesteps/save_interval)):
    model.learn(total_timesteps=save_interval, tb_log_name=name,
                log_interval=10, reset_num_timesteps=False)
    model.save("/media/ryuga/TOSHIBA EXT/BA/Models/" + name + "_" + str(i))
    dyn_lr.count()
    i += 1

# else:
#     while(i <= (timesteps/lr_update_interval)):
#         # linear: model.learning_rate = lr_start-(lr_stepsize*(i+pretraining_iterations))
#         # log:
#         model.learning_rate = lr_start*0.5**((i*lr_update_interval)*(10/timesteps))
#         # static:
#         #model.learning_rate = my_learning_rate
#         model.learn(total_timesteps=lr_update_interval, tb_log_name=name,
#                     log_interval=10, reset_num_timesteps=False)
#         if(i%40==39):
#             model.save("/media/ryuga/TOSHIBA EXT/BA/Models/" + name + "_" + str(i))
#         i += 1
