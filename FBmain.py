import torch
import os
import pickle
import json
import csv
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import pygame
from pygame.constants import K_w
import params
import utils
import models
from matplotlib import pyplot as plt



hparams = params.get_hparams()
#check if a CUDA GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': hparams.workers, 'pin_memory': True} if torch.cuda.is_available() else {}

#if no CUDA GPU available, check for Apple M1-specific GPU called MPS
if not torch.cuda.is_available():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.backends.mps.is_available():
        gpu='MPS'
    else:
        gpu=False
else:
    if torch.cuda.is_available():
        gpu='CUDA'
    else:
        gpu=False

lastAct = "Sig" if hparams.sigmoid else "Linear"
lastAct = "Softmax" if hparams.softmax else lastAct

#specified in ple/__init__.py lines 187-194
WIDTH = hparams.screenWidth     #if downsample by 4, w=72
HEIGHT = hparams.screenHeight    #if downsample by 4, h=100
GRID_SIZE = WIDTH * HEIGHT
ACTION_MAP = {'flap': K_w,'noop': None}
REWARDDICT = {"positive":1, "loss":-1}
OUTPUT = 2 if hparams.softmax else 1
PATH = hparams.output_dir + hparams.model_type +  f"-forwardDiscountScalar-greedy-fullFrame"
PATH, STATS, FRAMES = utils.build_directories(PATH)

with open(os.path.join(PATH,'output.txt'),'w') as f:
    f.write(f'gpu:{gpu}\n')
print(f'gpu:{gpu}')

rng = torch.Generator()
rng.manual_seed(hparams.seed)

#save metadata for easy viewing 
with open(PATH+'/metadata.txt', 'w') as f:
    json.dump(hparams.__dict__, f, indent=2)

#if environment does not support image rendering, this turns it off
if not hparams.render:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

###### Function definitions
def train(hparams, model,game):
    opt = torch.optim.Adam(params=model.parameters(),
                           lr=hparams.learning_rate,
                           maximize=hparams.maximize,
                           weight_decay=hparams.L2
                           )
    opt.zero_grad()
    
    #Initialize FB environment   
    game.init()
    WIDTH = 288
    HEIGHT = 400    

    training_summaries = []
    best_score, best_episode = -1,0
    
    with open(os.path.join(PATH,'output.txt'),'a') as f:
        f.write(f'commencing training with {hparams.model_type} model\n')
    print(f'commencing training with {hparams.model_type} model',flush=True)
    
    #train for num_episodes
    for episode in range(1,1+hparams.num_episodes):
        game.reset_game()
        
        agent_score, num_pipes = 0, 0
        frames, actions, rewards, probs = [], [], [], []

        lastFrame = np.zeros([WIDTH,HEIGHT],dtype=float)
        
        #play a single game
        while not game.game_over():
            #retrieve current game state and process into tensor
            currentFrame = game.getScreenGrayscale()
            frame_np = currentFrame[:,:400]
            # frame_np = utils.processScreen(currentFrame,WIDTH,HEIGHT)[0,:,:]
            
            #choose the appropriate model and get our action
            if 'NET' in hparams.model_type.upper():
                combined_np = np.subtract(frame_np,lastFrame).ravel()          #combine the current frame with the last frame, and flatten
                frame_t = torch.from_numpy(combined_np).float().to(DEVICE)     #convert to a tensor
                p = model(frame_t)
            elif 'CNN' in hparams.model_type.upper():
                stack_t = torch.stack([torch.from_numpy(frame_np).float(),torch.from_numpy(lastFrame).float()],0).to(DEVICE)
                p = model(stack_t)
            else:
                raise Exception('Unsupported model type.')         

            #update last frame array
            lastFrame = np.copy(frame_np)

            #get the action to take
            p_up = p[0].clone().to(DEVICE) if hparams.softmax else p.clone().to(DEVICE)
            # if gpu=="MPS":
            #     sample = torch.rand(1,dtype=torch.float32,generator=rng).to(DEVICE)
            # else:
            #     sample = torch.rand(1,dtype=float,generator=rng).to(DEVICE)
            # action = ACTION_MAP['flap'] if sample <= p_up else ACTION_MAP['noop'] 
            action = ACTION_MAP['flap'] if .5 <= p_up else ACTION_MAP['noop']   
            
            #take the action
            reward = game.act(action)
            agent_score += reward
            
            #record data for this step
            if reward > 0:
                num_pipes += 1
            frames.append(currentFrame)
            actions.append(1 if action==K_w else 0) #flaps stored as 1, no-flap stored as 0
            rewards.append(reward)
            probs.append(p_up)

        #end of game play for episode
            
        #update performance variables
        if num_pipes > best_score:
            string = ''
            best_score = num_pipes
            best_episode = episode
            with open(os.path.join(PATH,'output.txt'),'a') as f:
                f.write(f'\n{string}new high score:{best_score} episode:{best_episode}\n')
        training_summaries.append( (episode, num_pipes) )
        
        stacked_rewards = np.vstack(rewards)
        discounted_reward = torch.tensor(utils.discount_rewards(stacked_rewards, hparams.gamma)).float().to(DEVICE) 
        
        #calculate loss and do backprop
        prob_t = torch.stack(probs)         #create tensor of network outputs while preserving computational graph
        logp = torch.log(prob_t)            #add all log probabilities in the episode
        loss = torch.div(torch.mul(discounted_reward,logp), hparams.batch_size) #divide by number of samples (i.e. episodes in batch)
        
        try:
            loss.backward()
        except RuntimeError:
            loss.sum().backward()

        #accumulate gradient over batch_size episodes
        if episode % hparams.batch_size == 0:
            opt.step()
            opt.zero_grad()
        

        if episode % hparams.save_stats == 0:
            #save agent scores and episodes
            with open(STATS, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(training_summaries)
            training_summaries = []
            
    #save agent scores and episodes for final time
    with open(STATS, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(training_summaries)
        
    return model, (best_score, best_episode)



def evaluate(hparams, model,game):
    #Initialize FB environment   
    game.init()

    with open(os.path.join(PATH,'output.txt'),'a') as f:
        f.write(f'commencing evaluation\n')
    print(f'commencing evaluation',flush=True)

    game.reset_game()
        
    num_pipes = 0
    frames = []
    lastFrame = np.zeros([HEIGHT,WIDTH],dtype=float)
    f = 0    
    #play a single game
    while not game.game_over():
        f+=1
        #retrieve current game state and process into tensor
        currentFrame = game.getScreenRGB()
        frame_np = utils.processScreen(currentFrame,WIDTH,HEIGHT)[0,:,:]
        plt.imsave(os.path.join(FRAMES,f'evaluate_{f}.png'),currentFrame)
        
        #choose the appropriate model and get our action
        if 'NET' in hparams.model_type.upper():
            combined_np = np.subtract(frame_np,lastFrame).ravel()          #combine the current frame with the last frame, and flatten
            frame_t = torch.from_numpy(combined_np).float().to(DEVICE)     #convert to a tensor
            p = model(frame_t)
        elif 'CNN' in hparams.model_type.upper():
            stack_t = torch.stack([torch.from_numpy(frame_np).float(),torch.from_numpy(lastFrame).float()],0).to(DEVICE)
            p = model(stack_t)
        else:
            raise Exception('Unsupported model type.')         

        #update last frame array
        lastFrame = np.copy(frame_np)

        #get the action to take
        p_up = p[0].clone().to(DEVICE) if hparams.softmax else p.clone().to(DEVICE)
        action = ACTION_MAP['flap'] if .5 <= p_up else ACTION_MAP['noop']   
            
        #take the action
        reward = game.act(action)
            
        #record data for this step
        if reward > 0:
            num_pipes += 1
            frames.append(currentFrame)
    return num_pipes

    

#############   Main
FLAPPYBIRD = FlappyBird(rngSeed=hparams.seed)
game = PLE(FLAPPYBIRD, display_screen=hparams.render, force_fps=True, rng=hparams.seed, reward_values=REWARDDICT)


#train a model
if 'NET' in hparams.model_type.upper():
    model = models.PGNetwork(hparams,GRID_SIZE,OUTPUT,DEVICE).to(DEVICE)
elif 'CNN' in hparams.model_type.upper():
    model = models.CNN_PG(hparams, w=288, h=400, outputSize=OUTPUT,DEVICE=DEVICE).to(DEVICE)
else:
    raise Exception('Unsupported model type.')  

#evaulate the model resulting from full training length
lastModel, (best_score, best_episode) = train(hparams,model,game)
with open(os.path.join(PATH,'output.txt'),'a') as f:
    f.write(f'\ntraining completed\nbest score: {best_score} achieved at episode {best_episode}\n\nbeginning evaluation\n')
print(f'\ntraining completed\nbest score: {best_score} achieved at episode {best_episode}\n\nbeginning evaluation\n',flush=True)
num_pipes = evaluate(hparams,lastModel,game)
with open(os.path.join(PATH,'output.txt'),'a') as f:
        f.write(f'\nlast model evaluation completed\nscore: {num_pipes}\n')

# #attempt to evaluate the best overall model from training
# try:
#     bestModel = torch.load(PATH+'/bestModel.pt',map_location=DEVICE)
#     num_pipes = evaluate(hparams,bestModel,game)
#     with open(os.path.join(PATH,'output.txt'),'a') as f:
#         f.write(f'\ntrue best evaluation completed\nscore: {num_pipes}\n')
# except:
#     with open(PATH+'/digest.txt','w') as f:
#         f.write(f'Unable to load best model\n')

