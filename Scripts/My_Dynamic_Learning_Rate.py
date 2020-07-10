class LogLearningRate():

    """
    Logarithmically declining learning rate. This is a somewhat robust learning rate since it can apply learning rates of different orders of magnitude during one training, guaranteeing to hit the optimal static learning rate at some point.
    To make this learning rate even more robust at the cost of efficiency, chose a higher starting value (like 0.01) and a lower minimum (like 0.00001).
    If you know upper and lower bounds of a sensible learning rate (those are dependant on environment and algorithm!), plug them in and adapt the half life accordingly.
    Default values are a compromise between robustness and efficiency.

    :param timesteps: (int) It is recommended to put in the number of steps you plan to train your agent for, though any number can be chosen here.
    :param lr_start: (float) the starting value of the learning rate.
    :param lr_min: (float) clips the learning rate to a minimum value. Set to 0 for no clipping.
    :param half_life: (float) The (relative) fraction of timesteps after which the learning rate will reach half its initial value. E.g. 0.1 = 10%, Meaning half_life is 100,000 steps when training for 1,000,000 steps
    """
    
    def __init__(self, timesteps, lr_start = 0.001, lr_min = 0.00001, half_life = 0.1):
        self.timesteps = timesteps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.half_life_factor = 1/half_life

    def value(self, step):
        #print(str(step))
        s = (self.timesteps-(step*self.timesteps))
        lr = self.lr_start*0.5**(s*(self.half_life_factor/self.timesteps))
        #print("LR: " + str(lr))
        if(lr>self.lr_min):
            return lr
        else:
            return self.lr_min

