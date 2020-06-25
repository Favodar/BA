import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule
from NEW_FrankaGymEnvironment_DiscreteActions import CustomEnv
from stable_baselines.gail import generate_expert_traj

""" History
filename = "cFrom50k_CD5_Phys002_constLR_ppo2_franka_discrete_LR0.00075_timesteps_11200000srate_sreps_slimit_1002518"
model_iteration = "_6"
name = "cFrom120k_nightly_CD5_Phys002_constLR_ppo2_franka_discrete_LR"

filename = "Continued_CD6_LoadedFrom250kSteps_Phys002_ppo2_franka_discrete_LR_0.005-0.0001_timesteps_100000srate_sreps_slimit_1002512"
model_iteration = "_1"
name = "CD6_1.5xSlimit_staticLR_LoadedFrom270kSteps_Phys004_ppo2_franka_discrete_LR_"
"""

filename = "From510k_daily_CD6_Phys004_constLR_ppo2_franka_discrete_LR0.00075_timesteps_11200000srate_sreps_slimit_1002518"
model_iteration = "_6"
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

model = PPO2.load("/media/ryuga/Shared Storage/Models/" +
                  filename + model_iteration, env=env)
model.tensorboard_log = "/media/ryuga/Shared Storage/TensorBoardLogs/NEW_DEEP_FRANKA2"


# Demonstration of the loaded model
for i in range(1):
    obs = env.reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if(dones):
            break

print("NEW_PRETRAIN_DEEP_LOADED_PPO2_Franka.py LESS GO")

my_learning_rate = 0.00075  # scheduler.value default: 2.5e-4=0.00025
model.learning_rate = my_learning_rate

generate_expert_traj(model, save_path = 'pretrain_franka1', env = env,n_timesteps = 0, n_episodes=100)
