import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule
from NEW_Efficient_FrankaGymEnvironment_DiscreteActions_Revisited import CustomEnv

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
filename = "Rev_i7_DISCRETE_DefNN_RndmBall_Phys006_ppo2_franka_LR_0.00015-6.8e-05_timesteps_1000000_srate_sreps_slimit_1002580_joints_2_rdmBall_True_ballPos_[0.8, 0.8, 0]"
f = open("../Envparameters/envparameters_" + filename, "r")
envparameters = f.read()
envparameters = envparameters.strip('[')
envparameters = envparameters.strip(']')
f_list = [i for i in envparameters.split(",")]
print(
    "envparameters loaded from file [my_signal_rate, my_signal_repetitions, my_step_limit, lr_start, lr_end, timesteps]:")
print(str(f_list))

my_signal_rate = int(f_list[0])
my_signal_repetitions = int(f_list[1])
my_step_limit = int(f_list[2])
my_timesteps = -1
my_number_of_joints = 7
my_randomBall = True

try:
    my_timesteps = int(f_list[5])
    my_number_of_joints = int(f_list[6])
    my_randomBall = ((f_list[7]) == ' True')
except:
    print("timesteps, number_of_joints and randomBall params couldnt be loaded and were set to default vaules.")
    pass

print("this agent was aiming for a training with " + str(my_timesteps) + " timesteps.")
print("random Ball LOADED VALUE" + str(my_randomBall))
# Initialize environment with signal parameters:
env = CustomEnv(signal_rate= my_signal_rate, signal_repetitions= my_signal_repetitions, step_limit= my_step_limit, number_of_joints=my_number_of_joints, randomBall=my_randomBall)

# Load trained model and execute it forever:

# DELETE THIS BELOW
#!
filename += "_959"
#!
# DELETE THE ABOVE

model = PPO2.load("/media/ryuga/Shared Storage/Models/"+filename)

obs = env.reset()
env.render()
    
while True:
    obs = env.reset()
    for i in range(my_step_limit):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if(dones):
            break
