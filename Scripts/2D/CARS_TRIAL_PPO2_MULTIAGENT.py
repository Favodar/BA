import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from CARS_TRIAL_GymEnvironment_DiscreteActions import CustomEnv
from CARS_TRIAL_GymEnvironment_MULTIAGENT import CustomEnv as MultiEnv
from stable_baselines.common.schedules import ConstantSchedule, LinearSchedule

timesteps2 = 100000 #attacker_step_limit*100
scheduler = LinearSchedule(timesteps2, 0.001, 0.0001)
my_learning_rate2 = scheduler.value # 0.0005 

attacker_step_limit = 200
attacker_turnrate = 0.01745*(11.25/2)
attacker_maxspeed = 2.5
attacker_acceleration = 2.5/2
attacker_randomBall = True
attacker_binaryReward = True


   

# Initialize stub environment to break vicious circle
stub_env = MultiEnv(randomBall= attacker_randomBall, step_limit=attacker_step_limit, step_size = 0, maxspeed = 0,acceleration=0, binaryReward= attacker_binaryReward, isEscaping= False, enemy_model = 0, enemy_step_limit= 0, enemy_step_size= 0, enemy_maxspeed= 0, enemy_acceleration= 0)



print("CARS_PPO2_MULTIAGENT.py LESS GO")

defender_step_limit = attacker_step_limit
defender_turnrate = attacker_turnrate*2
defender_maxspeed = attacker_maxspeed*1.2
defender_acceleration = attacker_acceleration*0.9
defender_randomBall = True
defender_binaryReward = True

attacker_model = PPO2(MlpPolicy, stub_env, learning_rate= my_learning_rate2, verbose=1, tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/CARS_MULTI_NEW")

defender_env = MultiEnv(randomBall=defender_randomBall, step_limit=defender_step_limit, step_size = defender_turnrate, maxspeed = defender_maxspeed,acceleration=defender_acceleration, binaryReward= defender_binaryReward, isEscaping= True, enemy_model = attacker_model, enemy_step_limit= attacker_step_limit, enemy_step_size= attacker_turnrate, enemy_maxspeed= attacker_maxspeed, enemy_acceleration= attacker_acceleration)



defender_name = "CARS_NIGHT_ESCAPER_ppo2_LR_"  + "LinearSchedule_"  + "timesteps_" + str(timesteps2) + "ep_length_" + str(defender_step_limit) + "turnrate_" + str(defender_turnrate) + "maxspeed_" + str(defender_maxspeed) + "randomBall_" + str(defender_randomBall) + "binaryReward_" + str(defender_binaryReward)
attacker_name = "CARS_NIGHT_ATTACKER_ppo2_LR_"  + "LinearSchedule_"  + "timesteps_" + str(timesteps2) + "ep_length_" + str(attacker_step_limit) + "turnrate_" + str(attacker_turnrate) + "maxspeed_" + str(attacker_maxspeed) + "randomBall_" + str(attacker_randomBall) + "binaryReward_" + str(attacker_binaryReward)
 
defender_model = PPO2(MlpPolicy, defender_env, learning_rate= my_learning_rate2, verbose=1, tensorboard_log="/home/ryuga/Documents/TensorBoardLogs/CARS_MULTI_NEW")

attacker_env = MultiEnv(randomBall= attacker_randomBall, step_limit=attacker_step_limit, step_size = attacker_turnrate, maxspeed = attacker_maxspeed,acceleration=attacker_acceleration, binaryReward= attacker_binaryReward, isEscaping= False, enemy_model = defender_model, enemy_step_limit= defender_step_limit, enemy_step_size= defender_turnrate, enemy_maxspeed= defender_maxspeed, enemy_acceleration= defender_acceleration)
attacker_env = DummyVecEnv([lambda: attacker_env])
attacker_model.set_env(attacker_env)



try:
    f = open("../Envparameters/envparameters_" + attacker_name, "x")
    f.write(str([attacker_step_limit, attacker_turnrate, attacker_maxspeed, attacker_acceleration, attacker_randomBall, attacker_binaryReward]))
    f.close()
    f2 = open("../Envparameters/envparameters_" + defender_name, "x")
    f2.write(str([defender_step_limit, defender_turnrate, defender_maxspeed, defender_acceleration, defender_randomBall, defender_binaryReward]))
    f2.close()
except:
    print("envparameters couldn't be saved. They are:" + str([attacker_step_limit, attacker_turnrate, attacker_maxspeed, attacker_acceleration, attacker_randomBall, attacker_binaryReward]))

# Init learning
attacker_model.learn(total_timesteps=5000, tb_log_name= attacker_name + "INIT")



for i in range(10000):
    defender_model.learn(total_timesteps=timesteps2, tb_log_name= defender_name + str(i), log_interval=100)
    attacker_model.learn(total_timesteps= timesteps2, tb_log_name= attacker_name + str(i), log_interval=100)

    if(i%10==0):
        attacker_model.save("../Models/" + attacker_name+str(i))
        defender_model.save("../Models/" + defender_name+str(i))
        
    if(False):
        for i in range(3):
            obs = defender_env.reset()
            for i in range(attacker_step_limit):
                action, _states = defender_model.predict(obs)
                print(action)
                obs, rewards, dones, info = defender_env.step(action)
                defender_env.renderSlow(100)
                if(dones):
                    defender_env.renderSlow(1)
                    break
    

while True:
    obs = defender_env.reset()
    for i in range(attacker_step_limit):
        action, _states = defender_model.predict(obs)
        obs, rewards, dones, info = defender_env.step(action)
        defender_env.renderSlow(25)
