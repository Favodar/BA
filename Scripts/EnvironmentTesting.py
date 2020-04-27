from FrankaGymEnvironment import CustomEnv
from FrankaGymRewardNode2 import PublisherNode, Singleton, GymReward

if __name__ == '__main__':
    print("main says hi")
    env = CustomEnv()
    print("environment initialized")
    print(env.step([1,1]))

    # node = PublisherNode()

    # reward = GymReward()
    # reward.initializeNode()
    # print reward.getObservation([0,0,0,0,0,0,0])
    # print reward.getReward()
    
    # x = Singleton([3, 1])

    # print x.get_list
    # print x.get_list()
    # print x.get_list()[0]

    # #x.initializeNode()

    # # node = PublisherNode()
    # # node.initializeNode()
    # y = Singleton([2,2])

    # print y.get_list
    # print y.get_list()
    # print y.get_list()[0]


    # print x.get_list
    # print x.get_list()
    # print x.get_list()[0]
    # #print node.getReward()



