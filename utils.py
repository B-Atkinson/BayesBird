from inspect import currentframe
import numpy as np
import os
import pathlib

def build_directories(PATH):
    '''Builds all the directories needed to save test results to disk. Returns the paths to the various directories'''
    #create directory to save results to
    try:
        os.makedirs(PATH,exist_ok=False)
    except FileExistsError:
        #create a unique name for the directory in case of overlapping paths
        print('directory already exists:',PATH)
        from time import time
        PATH += "-"+str(time())[-5:]

    os.makedirs(os.path.join(PATH,'frames'),exist_ok=True)
    FRAMES = os.path.join(PATH,'frames')
    STATS = PATH+"/stats.csv"
    os.makedirs(os.path.dirname(PATH+'/metadata.txt'), exist_ok=True)
    print('Saving to: ' + PATH,flush=True)
    return PATH, STATS, FRAMES

def discount_rewards(r, gamma, disc_vector):
    '''This function analyzes the reward received after each action in an episode. It takes in a numpy array of rewards
    and applies the discount factor, resetting the decay counter each time a non-zero reward is received. In  doing  so 
    the network will view the sequence of actions between pipes as independent, which means that an agent  who  crossed
    3 pipes in an episode before dying would see 3 examples of successful strategies and  one unsuccessful  one  (as it
    died en-route to pipe 4).'''
    disc_r = np.zeros_like(r, dtype=float)
    running_sum = 0
    num_sets = np.core.numeric.count_nonzero(r)
    for t in reversed(range(0, len(r))):
        if r[t] == -1:  # If the reward is -1...
            running_sum = 0  # ...then reset sum, since it's a game boundary
        running_sum = running_sum * gamma + r[t]
        disc_r[t] = running_sum

    if disc_vector:
        #return a discounted reward array and the number of pipes
        return disc_r, num_sets
    else:
        #return a single scalar discounted reward and the number of pipes
        return sum(disc_r), num_sets