import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from CARS_TRIAL_GymEnvironment_DiscreteActions import CustomEnv
from My_Dynamic_Learning_Rate import LogLearningRate
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

my_step_limit = 120
my_step_size = 0.01745*22.5  # 0.01745*11.25
my_maxspeed = 2.5
my_acceleration = 2.5/4
my_randomBall = True
my_binaryReward = True

print("CARS_PPO2_DISCRETE.py LESS GO")

env = CustomEnv(step_limit=my_step_limit, step_size = my_step_size, maxspeed = my_maxspeed,acceleration=my_acceleration, randomBall = my_randomBall, binaryReward= my_binaryReward) # 0.01745*5
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
timesteps = 2000000

lr_start = 0.01 # macht erst was bei 0.00014
lr_end = 0.00001
half_life = 0.05
dyn_lr = LogLearningRate(timesteps= timesteps, lr_start = lr_start, lr_min = lr_end, half_life = half_life)

llr = LinearSchedule(timesteps, 0.005, 0.0001)  # default: 0.00025

my_learning_rate = 0.000063  # dyn_lr.value  # 0.000063
# my_learning_rate = scheduler.value
# my_learning_rate = 0.00075  # scheduler.value default: 2.5e-4=0.00025
#print_LR = str(lr_start) + "-" + str(lr_end)
print_LR = str(my_learning_rate)


#static_learning_rate = 0.00014  # my_learning_rate.value

#CRAZYDEEP7:
#p_quarks = dict(net_arch=[8192, 8192, dict(
#    vf=[8192, 4096, 4096, 2048], pi=[256, 256, 128])])
#CRAZYDEEP7 Lite:
#p_quarks = dict(net_arch=[4096, 4096, dict(
#    vf=[4096, 2048, 2048, 1024], pi=[256, 256, 128])])
#CRAZYDEEP7 SuperLite:
#p_quarks = dict(net_arch=[1024, 1024, dict(
#    vf=[1024, 512, 512, 256], pi=[256, 256, 128])])
#ReasonableDeep1:
p_quarks = dict(net_arch=[dict(
    vf=[128, 128, 128], pi=[128, 128, 128])])


name = "CARS_RD1_real22.5_staticLR_yesRender_medium5_225_newObs_ppo2_LR_" + print_LR + "halflife_" +str(half_life) + "_timesteps_" + str(timesteps) + "ep_length_" + str(
    my_step_limit) + "turnrate_" + str(my_step_size) + "maxspeed_" + str(my_maxspeed)
# Use tensorboard to show reward over time etc

model = PPO2(MlpPolicy, env, learning_rate=my_learning_rate, verbose=1,
             tensorboard_log="/media/ryuga/Shared Storage/TensorBoardLogs/CARSTRIAL_NEW") 

#model = PPO2(MlpPolicy, env, policy_kwargs=p_quarks, learning_rate=my_learning_rate, verbose=1,
#             tensorboard_log="/media/ryuga/Shared Storage/TensorBoardLogs/CARSTRIAL_NEW")

i = 0
save_interval = 500000
model.learn(total_timesteps=timesteps, tb_log_name= name, log_interval= 10, reset_num_timesteps= False)
# while(i <= (timesteps/save_interval)):
#     # linear: model.learning_rate = lr_start-(lr_stepsize*(i+pretraining_iterations))
#     model.learn(total_timesteps=save_interval, tb_log_name=name,
#                 log_interval=10, reset_num_timesteps=False)
#     model.save("/media/ryuga/Shared Storage/Models/" + name + "_" + str(i))
#     i += 1

model.save("/media/ryuga/Shared Storage/Models/" + name)

try:
    f = open("../Envparameters/envparameters_" + name, "x")
    f.write(str([my_step_limit, my_step_size, my_maxspeed, my_acceleration, my_randomBall, my_binaryReward]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" + str([my_step_limit, my_step_size, my_maxspeed, my_acceleration, my_randomBall, my_binaryReward]))



while True:
    obs = env.reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.renderSlow(25)
