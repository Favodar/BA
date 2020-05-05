import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from Franka2DGymEnvironmentNOREPS import CustomEnv

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

filename = "TWOD_FIXEDBALL_franka_CONTINUOUS_ppo2_LR_0.00025stepsize_0.08725timesteps_1000000ep_length_400"

# Load signal parameters from file:
f = open("../Envparameters/envparameters_" + filename, "r")
envparameters = f.read()
envparameters = envparameters.strip('[')
envparameters = envparameters.strip(']')
f_list = [float(i) for i in envparameters.split(",")]
print(str(f_list))

my_step_limit = int(f_list[0])
my_step_size = f_list[1]
my_number_of_joints = int(f_list[2])

[my_step_limit, my_step_size, my_number_of_joints]


# Initialize environment with signal parameters:
env = CustomEnv(step_limit=my_step_limit, step_size= my_step_size, number_of_joints= my_number_of_joints)

# Load trained model and execute it forever:
model = PPO2.load("../Models/"+filename)

obs = env.reset()
env.render()
while(True):
    env.reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    