from inspect import currentframe
import numpy as np
import os
import matplotlib.pyplot as plt
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
        PATH += "_"+str(time())[-5:]

    os.makedirs(os.path.join(PATH,'frames'),exist_ok=True)
    FRAMES = os.path.join(PATH,'frames')
    STATS = PATH+"/stats.csv"
    os.makedirs(os.path.dirname(PATH+'/metadata.txt'), exist_ok=True)
    print('Saving to: ' + PATH,flush=True)
    return PATH, STATS, FRAMES

def discount_rewards(r, gamma):
    # This function performs discounting of rewards by going back
    # and punishing or rewarding based upon final outcome of episode
    # known as thesisDiscount
    disc_r = np.zeros_like(r, dtype=float)
    running_sum = 0
    for t in reversed(range(0, len(r))):
        if r[t] == -1:  # If the reward is -1...
            running_sum = 0  # ...then reset sum, since it's a game boundary
        running_sum = running_sum * gamma + r[t]
        disc_r[t] = running_sum

    # Note that we add eps in the rare case that the std is 0
    return sum(disc_r)

def processScreen(obs,mean,std):
    obs = (obs - mean) / std
    return obs

def frameBurnIn(game,hparams,model,WIDTH,HEIGHT,DEVICE,gpu,burnIn=3):
    from pygame.constants import K_w
    import torch
    ACTION_MAP = {'flap': K_w,'noop': None}
    frames = []
    for _ in range(burnIn):
        game.reset_game()
        lastFrame = np.zeros([WIDTH,HEIGHT],dtype=float)
        gameFrames = []
        while not game.game_over():
            currentFrame = game.getScreenGrayscale()
            frame_np = currentFrame[:,:400]
            frame_np = cv2.resize(frame_np, (HEIGHT,WIDTH))

            #choose the appropriate model and get our action
            if 'NET' in hparams.model_type.upper():
                combined_np = np.subtract(frame_np,lastFrame).ravel()          #combine the current frame with the last frame, and flatten
                frame_t = torch.from_numpy(combined_np).float().to(DEVICE)     #convert to a tensor
                p = model(frame_t)
            elif 'CNN' in hparams.model_type.upper():
                frame_t = torch.stack([torch.from_numpy(frame_np).float(),torch.from_numpy(lastFrame).float()],0).to(DEVICE)
                p = model(frame_t)
            else:
                raise Exception('Unsupported model type.') 
            #update last frame array
            lastFrame = np.copy(frame_np)
            #get the action to take
            p_up = p[0].clone().to(DEVICE) if hparams.softmax else p.clone().to(DEVICE)
            if gpu=="MPS":
                sample = torch.rand(1,dtype=torch.float32).to(DEVICE)
            else:
                sample = torch.rand(1,dtype=float).to(DEVICE)
            action = ACTION_MAP['flap'] if sample <= p_up else ACTION_MAP['noop'] 
            _ = game.act(action)
            gameFrames.append(frame_t.cpu().numpy())
        gameFrames.pop(0)      #throw away first frame because of initialization
        frames.extend(gameFrames)

    frames = np.array(frames,dtype=float)
    return np.mean(frames), np.std(frames)