import numpy as np
import os
import cv2

def build_directories(PATH):
    '''Builds all the directories needed to save test results to disk. Returns the paths to the various directories'''
    #create directory to save results to
    try:
        os.makedirs(PATH,exist_ok=False)
    except FileExistsError:
        #create a unique name for the directory in case of overlapping paths
        print('directory already exists:',PATH)
        from time import time
        PATH += "_"+str(time())[-4:]

    FRAMES = os.makedirs(os.path.join(PATH,'frames'),exist_ok=True)
    STATS = PATH+"/stats.csv"
    os.makedirs(os.path.dirname(PATH+'/metadata.txt'), exist_ok=True)
    print('Saving to: ' + PATH,flush=True)
    return PATH, STATS, FRAMES
    
# def discount_rewards(r, gamma):
#     """ take 1D float array of rewards and compute discounted reward. known as forwardDiscount"""
#     discounts = np.zeros(r.size,dtype=float)
#     for t in range(0, r.size):
#         #discounted reward at this step = (discount_factor * running_sum last step) + reward for this step
#         discounts[t] =  (gamma**t) * r[t]
#     #return sum of discounted reward vector
#     return sum(discounts)

def discount_rewards(r, gamma):
    # This function performs discounting of rewards by going back
    # and punishing or rewarding based upon final outcome of episode
    # known as thesisDiscount
    disc_r = np.zeros_like(r, dtype=float)
    running_sum = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:  # If the reward is -1...
            running_sum = 0  # ...then reset sum, since it's a game boundary
        running_sum = running_sum * gamma + r[t]
        disc_r[t] = running_sum

    # Note that we add eps in the rare case that the std is 0
    return disc_r

# def processScreen(obs):
#     '''Takes as input a 512x288x3 numpy ndarray and downsamples it twice to get a 100x72 output array. Usless background 
#        pixels were manually overwritten with 33 in channel 0 to be easier to detect in-situ. Rows 400-512 never change 
#        because they're the ground, so they are cropped before downsampling. To reduce the number of parameters of the model,
#        only using the 0th channel of the original image.'''
#     obs = obs[::2,:400:2,0]
#     obs = obs[::2,::2]
#     col,row =np.shape(obs)
#     for i in range(col):
#         for j in range(row):
#             #background pixels only have value on channel 0, and the value is 33
#             if (obs[i,j]==33):
#                 obs[i,j] = 0
#             elif (obs[i,j]==0):
#                 pass                
#             else:
#                 obs[i,j] = 1
#     return obs.astype(np.float)

def processScreen(obs,w,h):
    image = cv2.cvtColor(cv2.resize(obs, (w,h)), cv2.COLOR_BGR2GRAY)
    # _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)
