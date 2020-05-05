import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from CARS_TRIAL_GymEnvironment_DiscreteActions import CustomEnv
from CARS_TRIAL_GymEnvironment_MULTIAGENT import CustomEnv as MultiEnv
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

filename = "CARS_medium5_225_newObs_ppo2_LR_LinearSchedule_timesteps_4000000ep_length_120turnrate_0.1963125maxspeed_2.5randomBall_TruebinaryReward_True"
# Load signal parameters from file:
f = open("../Envparameters/envparameters_" + filename, "r")
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


for i in range(1):
    obs = env.reset()
    for i in range(attacker_step_limit):
        action, _states = attacker_model.predict(obs)
        print(action)
        obs, rewards, dones, info = env.step(action)
        env.renderSlow(50)
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

env2 = MultiEnv(randomBall=defender_randomBall, step_limit=defender_step_limit, step_size = defender_step_size, maxspeed = defender_maxspeed,acceleration=defender_acceleration, binaryReward= defender_binaryReward, isEscaping= True, enemy_model = attacker_model, enemy_step_limit= attacker_step_limit, enemy_step_size= attacker_step_size, enemy_maxspeed= attacker_maxspeed, enemy_acceleration= attacker_acceleration)
timesteps2 = 50000#attacker_step_limit*100
scheduler = LinearSchedule(timesteps2, 0.001, 0.0001)
my_learning_rate2 = scheduler.value # 0.0005 
defender_name = "CARS_LSTM_ESCAPER_ppo2_LR_"  + "LinearSchedule_"  + "timesteps_" + str(timesteps2) + "ep_length_" + str(defender_step_limit) + "turnrate_" + str(defender_step_size) + "maxspeed_" + str(defender_maxspeed) + "randomBall_" + str(defender_randomBall) + "binaryReward_" + str(defender_binaryReward)
attacker_name = "CARS_MLP_ATTACKER_ppo2_LR_"  + "LinearSchedule_"  + "timesteps_" + str(timesteps2) + "ep_length_" + str(attacker_step_limit) + "turnrate_" + str(attacker_step_size) + "maxspeed_" + str(attacker_maxspeed) + "randomBall_" + str(attacker_randomBall) + "binaryReward_" + str(attacker_binaryReward)
 
defender_model = PPO2(MlpLstmPolicy, env2, nminibatches=1, learning_rate= my_learning_rate2, verbose=1, tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/CARS_MULTI_RNN")

env = MultiEnv(randomBall= attacker_randomBall, step_limit=attacker_step_limit, step_size = attacker_step_size, maxspeed = attacker_maxspeed,acceleration=attacker_acceleration, binaryReward= attacker_binaryReward, isEscaping= False, enemy_model = defender_model, enemy_step_limit= defender_step_limit, enemy_step_size= defender_step_size, enemy_maxspeed= defender_maxspeed, enemy_acceleration= defender_acceleration)
env = DummyVecEnv([lambda: env])
attacker_model.set_env(env)
attacker_model.tensorboard_log = "/home/ryuga/Documents/TensorBoardLogs/CARS_MULTI_RNN"
attacker_model.learning_rate = my_learning_rate2

try:
    f = open("../Envparameters/envparameters_" + attacker_name, "x")
    f.write(str([attacker_step_limit, attacker_step_size, attacker_maxspeed, attacker_acceleration, attacker_randomBall, attacker_binaryReward]))
    f.close()
    f2 = open("../Envparameters/envparameters_" + defender_name, "x")
    f2.write(str([defender_step_limit, defender_step_size, defender_maxspeed, defender_acceleration, defender_randomBall, defender_binaryReward]))
    f2.close()
except:
    print("envparameters couldn't be saved. They are:" + str([attacker_step_limit, attacker_step_size, attacker_maxspeed, attacker_acceleration, attacker_randomBall, attacker_binaryReward]))

# Init learning
defender_model.learn(total_timesteps=500000, tb_log_name= defender_name + "INIT")
defender_model.save("../Models/" + defender_name+str(i))



for i in range(10000):
    defender_model.learn(total_timesteps=timesteps2, tb_log_name= defender_name + str(i), log_interval=100)
    attacker_model.learn(total_timesteps= timesteps2//2, tb_log_name= attacker_name + str(i), log_interval=100)

    if(i%3==0):
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
    

while True:
    obs = env2.reset()
    for i in range(attacker_step_limit):
        action, _states = defender_model.predict(obs)
        obs, rewards, dones, info = env2.step(action)
        env2.renderSlow(25)