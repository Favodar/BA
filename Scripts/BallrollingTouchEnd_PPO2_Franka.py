import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule
from BallrollingTouchEnd_FrankaGymEnvironment_ContinuousActions import CustomEnv
from My_Dynamic_Learning_Rate import ExpLearningRate

id = "BTE_"
my_signal_rate = 100
my_signal_repetitions = 25
my_step_limit = 100
my_number_of_joints = 2
my_randomBall = False
my_randomTarget = False
my_ballPos = [0.5, 0.5, 0]
my_steps_after_kick = 80

env = CustomEnv(signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit, number_of_joints= my_number_of_joints, randomBall= my_randomBall,randomTarget=my_randomTarget, ballPos = my_ballPos, steps_after_kick = my_steps_after_kick)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

timesteps = 100000
save_interval = 10000
is_static_lr = False

# for dynamic LRs:
lr_start = 0.001
lr_end = 0.000063
half_life = 0.1
dyn_lr = ExpLearningRate(
    timesteps=timesteps, lr_start=lr_start, lr_min=lr_end, half_life=half_life, save_interval=save_interval)
my_learning_rate = dyn_lr.value # 0.0005  # 0.000063 #scheduler.value # 0.0005 default: 2.5e-4=0.00025
#scheduler = LinearSchedule(schedule_timesteps= timesteps,initial_p= lr_start, final_p = lr_end)

#print_LR = str(my_learning_rate) 
print_LR = str(lr_start) + "-" + str(lr_end)

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

# CRAZYDEEP6:
p_quarks = dict(net_arch=[dict(
    vf=[1024, 1024, 1024, 1024, 1024], pi=[256, 256, 128])])

#CRAZYDEEP7:
#p_quarks = dict(net_arch=[8192, 8192, dict(
#    vf=[8192, 4096, 4096, 2048], pi=[256, 256, 128])])

#CRAZYDEEP7 Lite:
#p_quarks = dict(net_arch=[4096, 4096, dict(
#    vf=[4096, 2048, 2048, 1024], pi=[256, 256, 128])])

#CRAZYDEEP7 SuperLite:
#p_quarks = dict(net_arch=[1024, 1024, dict(
#    vf=[1024, 512, 512, 256], pi=[256, 256, 128])])

#ReasonableDeep2:
#p_quarks = dict(net_arch=[256, 256, dict(
#    vf=[256, 128], pi=[64])])


name = id + "Ryzen_SAK80_ELR_Jack01Fixed_CD6run2_Phys006_ppo2_franka_continuous_LR_" + print_LR + "_timesteps_" + \
    str(timesteps) + "_srate_sreps_slimit_" + str(my_signal_rate) + \
    str(my_signal_repetitions) + str(my_step_limit) + "_joints_" + str(my_number_of_joints) + "_rdm_" + str(my_randomBall) + "_ballPos_" + str(my_ballPos) + "_stepsAfterK_" + str(my_steps_after_kick)


#model = PPO2(MlpPolicy, env, policy_kwargs=p_quarks, learning_rate=my_learning_rate, verbose=1,
#             tensorboard_log="/media/ryuga/TOSHIBA EXT/BA/TensorBoardLogs/NEW_DEEP_FRANKA5_RYZEN")  # defaults: learning_rate=2.5e-4,

policy = MlpPolicy # if MlpLstmPolicy then nminibatches=1 # MlpPolicy
model = PPO2(policy, env, policy_kwargs=p_quarks, learning_rate=my_learning_rate, verbose=1,
             tensorboard_log="/media/ryuga/TOSHIBA EXT/BA/TensorBoardLogs/COMPARE_NNs_BALLROLL_TOUCHEND_RYZEN")  # defaults: learning_rate=2.5e-4,

try:
    f = open("../Envparameters/envparameters_" + name, "x")
    f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit, lr_start, lr_end, timesteps, my_number_of_joints, my_randomBall, my_ballPos, my_steps_after_kick]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" +
          str([my_signal_rate, my_signal_repetitions, my_step_limit, lr_start, lr_end, timesteps, my_number_of_joints, my_randomBall, my_ballPos, my_steps_after_kick]))

#print("Warning: default network architecture")

lr_update_interval = 500



lr_stepsize = (lr_start-lr_end)/(timesteps/lr_update_interval)
print("lr_stepsize: " + str(lr_stepsize))

model.learn(total_timesteps=500, tb_log_name=name,
            log_interval=10, reset_num_timesteps=False)
i = 0
if(is_static_lr):
    model.learning_rate = my_learning_rate
    while(i <= (timesteps/save_interval)):
        
        model.learn(total_timesteps=save_interval, tb_log_name=name,
                    log_interval=10, reset_num_timesteps=False)
        model.save("/media/ryuga/TOSHIBA EXT/BA/Models/" + name + "_" + str(i))
        i += 1

else:
    while(True):  # i <= (timesteps/save_interval)):
        model.learn(total_timesteps=save_interval, tb_log_name=name,
                    log_interval=10, reset_num_timesteps=False)
        model.save("/media/ryuga/TOSHIBA EXT/BA/Models/" + name + "_" + str(i))
        dyn_lr.count()
        i += 1




