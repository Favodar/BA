import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from CARS_TRIAL_GymEnvironment_DiscreteActions import CustomEnv
from CARS_TRIAL_GymEnvironment_MULTIAGENT import CustomEnv as MultiEnv
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

#filename = "CARS_medium5_225_newObs_ppo2_LR_LinearSchedule_timesteps_4000000ep_length_120turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True40"# "CARS_medium5_225_newObs_ppo2_LR_LinearSchedule_timesteps_4000000ep_length_120turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True"
filename = "CARS_NIGHT_ATTACKER_ppo2_LR_LinearSchedule_timesteps_100000ep_length_500turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True230"
# Load signal parameters from file:
#f = open("../Envparameters/envparameters_" + filename, "r")
f = open("../Envparameters/envparameters_CARS_ATTACKER_ppo2_LR_LinearSchedule_timesteps_30000ep_length_300turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True", "r")
envparameters = f.read()
envparameters = envparameters.strip('[')
envparameters = envparameters.strip(']')
f_list = [i for i in envparameters.split(",")]
print("envparameters: " + str(f_list))

attacker_step_limit = int(f_list[0])
attacker_step_size = float(f_list[1])
attacker_maxspeed = float(f_list[2])
attacker_acceleration = 2.5/4
attacker_randomBall = True
attacker_binaryReward = True


   

# Initialize environment with signal parameters:
env = CustomEnv(step_limit=attacker_step_limit, step_size = attacker_step_size, maxspeed = attacker_maxspeed, acceleration = attacker_acceleration, randomBall=attacker_randomBall, binaryReward=attacker_binaryReward) # 0.01745*5

# Load trained model and execute it forever:
attacker_model = PPO2.load("../Models/" +filename)

showAttackerPreview = False
if(showAttackerPreview):
    for i in range(10):
        obs = env.reset()
        for i in range(attacker_step_limit):
            action, _states = attacker_model.predict(obs)
            print(action)
            obs, rewards, dones, info = env.step(action)
            env.renderSlow(200)
            if(dones):
                env.renderSlow(1)
                break

print("CARS_PPO2_LOADED_MULTIAGENT.py LESS GO")

attacker_step_limit = 500

defender_step_limit = attacker_step_limit
defender_step_size = attacker_step_size
defender_maxspeed = attacker_maxspeed*0.9
defender_acceleration = attacker_acceleration*0.9
defender_randomBall = True
defender_binaryReward = True

print("Load_CARS_MULTI_PPO2.py LESS GO")

filename2 = "CARS_NIGHT_ESCAPER_ppo2_LR_LinearSchedule_timesteps_100000ep_length_500turnrate_0.1963125maxspeed_2.25randomBall_TruebinaryReward_True230"

defender_model = PPO2.load("../Models/" +filename2)
# Load signal parameters from file:
# f2 = open("../Envparameters/envparameters_" + filename2, "r")
# envparameters2 = f2.read()
# envparameters2 = envparameters.strip('[')
# envparameters2 = envparameters.strip(']')
# f_list2 = [i for i in envparameters2.split(",")]
# print("envparameters2: " + str(f_list2))

# my_step_limit2 = int(f_list2[0])
# my_step_size2 = float(f_list2[1])
# my_maxspeed2 = float(f_list2[2])
# my_acceleration2 = 2.5/4
# my_randomBall2 = True
# my_binaryReward2 = True

attacker_name = "FURTHER" + filename
defender_name = "FURTHER" + filename2
   

# Initialize environment with signal parameters:
env2 = MultiEnv(randomBall=defender_randomBall,step_limit=defender_step_limit, step_size = defender_step_size, maxspeed = defender_maxspeed,acceleration=defender_acceleration, binaryReward= defender_binaryReward, isEscaping= True, enemy_model = attacker_model, enemy_step_limit= attacker_step_limit, enemy_step_size= attacker_step_size, enemy_maxspeed= attacker_maxspeed, enemy_acceleration= attacker_acceleration)
env = MultiEnv(randomBall=attacker_randomBall,step_limit=attacker_step_limit, step_size = attacker_step_size, maxspeed = attacker_maxspeed,acceleration=attacker_acceleration, binaryReward= attacker_binaryReward, isEscaping= False, enemy_model = defender_model, enemy_step_limit= defender_step_limit, enemy_step_size= defender_step_size, enemy_maxspeed= defender_maxspeed, enemy_acceleration= defender_acceleration)

env = DummyVecEnv([lambda: env])
env2 = DummyVecEnv([lambda: env2])

attacker_model.set_env(env)
defender_model.set_env(env2)

timesteps2 = 500000

scheduler = LinearSchedule(timesteps2, 0.001, 0.0001)
my_learning_rate2 = scheduler.value



for i in range(10000):
    defender_model.learn(total_timesteps=timesteps2, tb_log_name= defender_name + str(i), log_interval=100)
    attacker_model.learn(total_timesteps= timesteps2//2, tb_log_name= attacker_name + str(i), log_interval=100)

    if(i%10==0):
        attacker_model.save("../Models/" + attacker_name+str(i))
        defender_model.save("../Models/" + defender_name+str(i))
        
    if(False):
        for i in range(3):
            obs = env2.reset()
            for i in range(attacker_step_limit):
                action, _states = defender_model.predict(obs)
                print(action)
                obs, rewards, dones, info = env2.step(action)
                env2.renderSlow(100)
                if(dones):
                    env2.renderSlow(1)
                    break