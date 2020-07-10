import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from CARS_TRIAL_GymEnvironment_DiscreteActions import CustomEnv
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

my_step_limit = 120
my_step_size = 0.01745*11.25
my_maxspeed = 2.5
my_acceleration = 2.5/4
my_randomBall = True
my_binaryReward = True

print("CARS_PPO2_DISCRETE.py LESS GO")

env = CustomEnv(step_limit=my_step_limit, step_size = my_step_size, maxspeed = my_maxspeed,acceleration=my_acceleration, randomBall = my_randomBall, binaryReward= my_binaryReward) # 0.01745*5
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
timesteps = 4000000

lr_start = 0.001 # macht erst was bei 0.00014
lr_end = 0.0001
scheduler = LinearSchedule(timesteps, lr_start, lr_end)
# my_learning_rate = scheduler.value
# my_learning_rate = 0.00075  # scheduler.value default: 2.5e-4=0.00025
print_LR = str(lr_start) + "-" + str(lr_end)
# print_LR = str(my_learning_rate)


static_learning_rate = 0.00014  # my_learning_rate.value

#CRAZYDEEP7:
p_quarks = dict(net_arch=[8192, 8192, dict(
    vf=[8192, 4096, 4096, 2048], pi=[256, 256, 128])])


name = "CARS_CD7_logStartEnd001_try2_4Mill_noRender_medium5_225_newObs_ppo2_LR_" + print_LR + "timesteps_" + str(timesteps) + "ep_length_" + str(
    my_step_limit) + "turnrate_" + str(my_step_size) + "maxspeed_" + str(my_maxspeed) + "randomBall_" + str(my_randomBall) + "binaryReward_" + str(my_binaryReward)
# Use tensorboard to show reward over time etc

#model = PPO2(MlpPolicy, env, learning_rate=static_learning_rate, verbose=1,
#             tensorboard_log="/media/ryuga/Shared Storage/TensorBoardLogs/CARSTRIAL") 

model = PPO2(MlpPolicy, env, policy_kwargs=p_quarks, learning_rate=static_learning_rate, verbose=1,
             tensorboard_log="/media/ryuga/Shared Storage/TensorBoardLogs/CARSTRIAL_NEW")
save_interval = 10000
pretraining_iterations = 0

lr_stepsize = (lr_start-lr_end)/(timesteps/save_interval)
#print("lr_stepsize: " + str(lr_stepsize))

model.learning_rate = lr_start

i = 0
while(i <= (timesteps/save_interval)):
    # linear: model.learning_rate = lr_start-(lr_stepsize*(i+pretraining_iterations))
    # log: 
    model.learning_rate = lr_end+(lr_start-lr_end)*0.5**((i*save_interval)*(10/timesteps))
    # model.learning_rate = static_learning_rate
    model.learn(total_timesteps=save_interval, tb_log_name=name,
                log_interval=10, reset_num_timesteps=False)
    #model.save("/media/ryuga/Shared Storage/Models/" + name + "_" + str(i))
    i += 1

model.save("/media/ryuga/Shared Storage/Models" + name)

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
