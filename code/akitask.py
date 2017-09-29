import numpy as np
import copy

class akitask():

    """
    The task consist in discriminating a single real value number against a threshold (zero).
    The task is organized in episodes (or blocks), each of which is composed by a random number of trials
    Blocks can have different prior probability of the correct response
    Blocks can also have different rewards for the response.
    """    

    def __init__(self,nTrials_limits,noise,priors,rewards):

        """
        Args: 
        nTrials_limits: low and high limit on the nr of trials for an episode
        noise: overall additive noise strengh for stimulus
        priors: list of side prior probability of each episode type
        rewards: high and low reward for correct action according to the episode type
        """
        self.num_actions = 2
        self.nTrials_limits = nTrials_limits
        self.nTrials = np.random.randint(self.nTrials_limits[0],self.nTrials_limits[1])
        self.h = np.random.randint(2)
        self.noise = noise
        self.priors = priors
        self.rewards = rewards
        self.reset()
        
    def get_state(self):
        
        # stimulus absolute strength
        r = np.random.uniform()*5/10.0
        
        # correct side is drawn from prior probabilities        
        if np.random.uniform() < self.priors[self.h]:
          s = -1
          self.correct_action = 0
        else:
          s = 1
          self.correct_action = 1
        
        # additional noise 
        n = np.random.randn()*self.noise        
                
        self.state = [0.5+s*(r+n)]

        return self.state, self.correct_action
                
    def reset(self):
        self.timestep = 0
        self.nTrials = np.random.randint(self.nTrials_limits[0],self.nTrials_limits[1])
        


        # Alternate: switch to next h
#       self.h += 1
#       self.h = self.h%len(self.priors)
          
        # Randomize hidden state
        self.h = np.random.randint(len(self.priors)) 



        return self.get_state()
        
    def pullArm(self,action):
        self.timestep += 1
        if self.correct_action == action:
            # reward depends on block
            # self.rewards[0] is high reward 
            # self.rewards[1] is low reward 
            if action == self.h:
              reward = self.rewards[0]
            else:
              reward = self.rewards[1]
        else:
            reward = 0.0

        new_state,correct_action = self.get_state()
        if self.timestep > self.nTrials: 
            done = True
        else: 
            done = False

        return new_state,reward,done,self.timestep,correct_action 

