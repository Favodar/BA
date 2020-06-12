import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule
from NEW_FrankaGymEnvironment_DiscreteActions import CustomEnv

# f = open("envparameters.txt", "r")
# my_list = f.read()
# my_signal_rate = my_list[0]
# my_signal_repetitions = my_list[1]
# my_step_limit = my_list[2]
# f.close()

# Standard:
# my_signal_rate = 100
# my_signal_repetitions = 15
# my_step_limit = 16

# Load signal parameters from file:
filename = "NEW_CD6_LoadedFrom60k+Steps_simplePhys01_newTryNightly_ppo2_franka_discrete_LR_0.00025_timesteps_1200000srate_sreps_slimit_1002512"
f = open("../Envparameters/envparameters_" + filename, "r")
envparameters = f.read()
envparameters = envparameters.strip('[')
envparameters = envparameters.strip(']')
f_list = [int(i) for i in envparameters.split(",")]
print(str(f_list))

my_signal_rate = f_list[0]
my_signal_repetitions = f_list[1]
my_step_limit = f_list[2]

# Initialize environment with signal parameters:
env = CustomEnv(signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit)

# Load trained model and execute it forever:

# DELETE THIS BELOW
#!
filename += "_6"
#!
# DELETE THE ABOVE

model = PPO2.load("../Models/"+filename)

obs = env.reset()
env.render()
    
while True:
    obs = env.reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if(dones):
            break
