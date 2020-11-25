import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule
from Ballrolling_FrankaGymEnvironment_DiscreteActions import CustomEnv

my_signal_rate = 100
my_signal_repetitions = 25
my_step_limit = 100
my_number_of_joints = 2
my_randomBall = False
my_ballPos = [0.5, 0.5, 0]

env = CustomEnv(signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit, number_of_joints= my_number_of_joints, randomBall= my_randomBall, ballPos = my_ballPos)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

timesteps = 3000000
is_static_lr = True
lr_start = 0.0005
lr_end = 0.000063
#scheduler = LinearSchedule(schedule_timesteps= timesteps,initial_p= lr_start, final_p = lr_end)
my_learning_rate = 0.0005 #0.000063 #scheduler.value # 0.0005 default: 2.5e-4=0.00025
print_LR = str(my_learning_rate) 
#print_LR = str(lr_start) + "-" + str(lr_end)

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

#ReasonableDeep2:
p_quarks = dict(net_arch=[256, 256, dict(
    vf=[256, 128], pi=[64])])


name = "Colloquium2_i7_LessPunishTry2_6xSTEPSIZE_RD2_Phys006_ppo2_franka_discrete_LR_" + print_LR + "_timesteps_" + \
    str(timesteps) + "_srate_sreps_slimit_" + str(my_signal_rate) + \
    str(my_signal_repetitions) + str(my_step_limit) + "_joints_" + str(my_number_of_joints) + "_rdmBall_" + str(my_randomBall) + "_ballPos_" + str(my_ballPos)


#model = PPO2(MlpPolicy, env, policy_kwargs=p_quarks, learning_rate=my_learning_rate, verbose=1,
#             tensorboard_log="/media/ryuga/Shared Storage/TensorBoardLogs/NEW_DEEP_FRANKA5_RYZEN")  # defaults: learning_rate=2.5e-4,

policy = MlpPolicy # if MlpLstmPolicy then nminibatches=1 # MlpPolicy
model = PPO2(policy, env, policy_kwargs=p_quarks, learning_rate=my_learning_rate, verbose=1,
             tensorboard_log="/media/ryuga/Shared Storage/TensorBoardLogs/COMPARE_FRANKA_BALLROLL")  # defaults: learning_rate=2.5e-4,

try:
    f = open("../Envparameters/envparameters_" + name, "x")
    f.write(str([my_signal_rate, my_signal_repetitions, my_step_limit, lr_start, lr_end, timesteps, my_number_of_joints, my_randomBall, my_ballPos]))
    f.close()
except:
    print("envparameters couldn't be saved. They are:" +
          str([my_signal_rate, my_signal_repetitions, my_step_limit, lr_start, lr_end, timesteps, my_number_of_joints, my_randomBall, my_ballPos]))

#print("Warning: default network architecture")

lr_update_interval = 500

save_interval = 20000

lr_stepsize = (lr_start-lr_end)/(timesteps/lr_update_interval)
print("lr_stepsize: " + str(lr_stepsize))

i = 0

if(is_static_lr):
    while(i <= (timesteps/save_interval)):
        model.learning_rate = my_learning_rate
        model.learn(total_timesteps=save_interval, tb_log_name=name,
                    log_interval=10, reset_num_timesteps=False)
        model.save("/media/ryuga/Shared Storage/Models/" + name + "_" + str(i))
        i += 1

else:
    while(i <= (timesteps/lr_update_interval)):
        # linear: model.learning_rate = lr_start-(lr_stepsize*(i+pretraining_iterations))
        # log:
        lr_now = lr_start*0.5**((i*lr_update_interval)*(10/timesteps))
        if(lr_now<lr_end):
            lr_now = lr_end
        model.learning_rate = lr_now
        # static:
        #model.learning_rate = my_learning_rate
        model.learn(total_timesteps=lr_update_interval, tb_log_name=name,
                    log_interval=10, reset_num_timesteps=False)
        if(i%40==39):
            model.save("/media/ryuga/Shared Storage/Models/" + name + "_" + str(i))
        i += 1




