import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from FrankaGymEnvironment import CustomEnv

my_signal_rate = 100
my_signal_repetitions = 25
my_step_limit = 50

env = CustomEnv(signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

timesteps = 2000000
lr_start = 0.001
lr_end = 0

my_learning_rate = 0.003

print_LR = str(lr_start) + "-" + str(lr_end)

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

print("IMPORTANT: Using old, less efficient observation-calculator for comparsion purposes.")
print("The import should be changed to the NEW_Efficient_FrankaGymRewardNode3RandomBall.py in the future!")

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise,
             tensorboard_log="/media/ryuga/TOSHIBA EXT/BA/TensorBoardLogs/NEW_DEEP_FRANKA5_RYZEN")

name = "RYZEN_DDPG_withNoise_NewEnvParams_DefaultNN_LogLR_Phys004_ddpg_franka_continuous_LR_" + print_LR + "_timesteps_" + \
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


lr_update_interval = 500
model_save_interval = 20000
modulo_number = model_save_interval/lr_update_interval

i = 0
while(i <= (timesteps/lr_update_interval)):
    # linear: model.learning_rate = lr_start-(lr_stepsize*(i+pretraining_iterations))
    # log:
    model.learning_rate = lr_start*0.5**((i*lr_update_interval)*(10/timesteps))
    # static: model.learning_rate = static_learning_rate
    model.learn(total_timesteps=lr_update_interval, tb_log_name=name,
                log_interval=10, reset_num_timesteps=False)
    if(i % modulo_number == (modulo_number-1)):
        model.save("/media/ryuga/TOSHIBA EXT/BA/Models/" +
                   name + "_" + str(i/modulo_number))
    i += 1
