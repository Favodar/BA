import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from FrankaGymEnvironmentEASY import CustomEnv

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

filename = "EZBALLS_franka_continuous_ppo20.003_timesteps_160000"

# Load signal parameters from file:
f = open("envparameters_" + filename, "r")
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
model = PPO2.load(filename)

obs = env.reset()
env.render()
while(True):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    