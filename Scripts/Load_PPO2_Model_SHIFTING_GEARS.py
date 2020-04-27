import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from FrankaGymEnvironment_ShiftingGears import CustomEnv

# Standard:
# my_signal_rate = 100
# my_signal_repetitions = 15
# my_step_limit = 16

# Shifting Gears:
# my_signal_rate = 100
# my_signal_repetitions = 2
# my_step_limit = 100

filename = "BALLS_ppo2_franka_SHIFTING_GEARS_learning_rate_0.003_sps_6.666666666666667_timesteps_40000"

# Load signal parameters from file:
f = open("envparameters_" + filename, "r")
envparameters = f.read()
envparameters = envparameters.strip('[')
envparameters = envparameters.strip(']')
f_list = [float(i) for i in envparameters.split(",")]
print("envparameters: " + str(f_list))

my_signal_rate = int(f_list[0])
my_signal_repetitions = int(f_list[1])
my_step_limit = int(f_list[2])
try:
    my_number_of_gears = int(f_list[3])
except IndexError:
    my_number_of_gears = 2
    print("number of gears couldn't be loaded from file and was defaulted to 2")
try:
    my_gear_interval = f_list[4]
except IndexError:
    my_gear_interval = 0.05
    print("gear interval couldn't be loaded from file and was defaulted to 0.05")
   

# Initialize environment with signal parameters:
env = CustomEnv(number_of_gears= my_number_of_gears, signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit, gear_interval=my_gear_interval)

# Load trained model and execute it forever:
model = PPO2.load(filename)

obs = env.reset()
env.render()
while(True):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    