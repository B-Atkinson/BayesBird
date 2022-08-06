import torch
import torch.nn.functional as F
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
from tqdm import tqdm



hparams = params.get_hparams()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': hparams.workers, 'pin_memory': True} if torch.cuda.is_available() else {}
print(f'use gpu:{torch.cuda.is_available()}   device:{DEVICE}')

#specified in ple/__init__.py lines 187-194
WIDTH = 100     #downsample by half twice
HEIGHT = 72    #downsample by half twice
GRID_SIZE = WIDTH * HEIGHT
ACTION_MAP = {'flap': K_w,'noop': None}
REWARDDICT = {"positive":2, "loss":-5}
OUTPUT = 1
PATH = hparams.output_dir + "-"+ hparams.model_type +  "-S" + str(hparams.seed)
PATH, STATS = utils.build_directories(hparams,PATH)

rng = torch.Generator()
rng.manual_seed(hparams.seed)

#save metadata for easy viewing 
with open(PATH+'/metadata.txt', 'w') as f:
    json.dump(hparams.__dict__, f, indent=2)

#if environment does not support image rendering, this turns it off
if not hparams.render:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

###### Function definitions
def train(hparams, model):
    opt = torch.optim.Adam(params=model.parameters(),
                           lr=hparams.learning_rate,
                           maximize=hparams.maximize,
                           weight_decay=hparams.L2
                           )
    
    #Initialize FB environment   
    FLAPPYBIRD = FlappyBird(rngSeed=hparams.seed)
    game = PLE(FLAPPYBIRD, display_screen=hparams.render, force_fps=True, rng=hparams.seed, reward_values=REWARDDICT)
    game.init()
    
    training_summaries = []
    best_score, best_episode = -1,0
    
    print(f'commencing training with {hparams.model_type} model')
    
    #train for num_episodes
    for episode in range(1,1+hparams.num_episodes):
        game.reset_game()
        
        agent_score, num_pipes = 0, 0
        frames, actions, rewards, probs = [], [], [], []
        lastFrame = np.zeros([GRID_SIZE])
        
        #play a single game
        while not game.game_over():
            #retrieve current game state and process into tensor
            currentFrame = game.getScreenRGB()
            frame_np = utils.processScreen(currentFrame)
            combined_np = np.subtract(frame_np,lastFrame)    #combine the current frame with the last frame, and flatten
            frame_t = torch.from_numpy(combined_np).float().to(DEVICE)     #convert to a tensor
            lastFrame = np.copy(frame_np)
            
            #choose the appropriate model and get our action
            if hparams.model_type == 'PGNetwork':
                p = model(frame_t)
                sample = torch.rand(1,dtype=float,generator=rng).to(DEVICE)
                action = ACTION_MAP['flap'] if sample < p else ACTION_MAP['noop']
            elif hparams.model_type == 'NoisyPG':
                p = model(frame_t)
                action = ACTION_MAP['flap'] if torch.rand(1,dtype=float,generator=rng) < p else ACTION_MAP['noop']
            else:
                raise Exception('Unsupported model type.')            
            
            #take the action
            reward = game.act(action)
            agent_score += reward
            
            #record data for this step
            if reward > 0:
                num_pipes += 1
            frames.append(frame_t)
            actions.append(1 if action==K_w else 0) #flaps stored as 1, no-flap stored as 0
            rewards.append(reward)
            probs.append(p)
        #end of game play for episode
            
        #update performance variables
        if num_pipes > best_score:
            pickle.dump(frames,open(PATH+'/bestFrames.p','wb'))
            # pickle.dump(model, open(MODEL_NAME  + str(episode) + '.p', 'wb'))
            best_score = num_pipes
            best_episode = episode
        training_summaries.append( (episode, num_pipes) )
        
        stacked_rewards = np.vstack(rewards)
        discounted_reward = torch.tensor(utils.discount_rewards(stacked_rewards, hparams.gamma)).float().to(DEVICE) 
        
        #calculate loss and do backprop
        if hparams.model_type == 'PGNetwork':
            prob_t = torch.stack(probs)             #create tensor of network outputs while preserving computational graph
            logp = torch.sum(torch.log(prob_t))     #add all log probabilities in the episode
            loss = torch.div(torch.mul(discounted_reward,logp), hparams.batch_size) #divide by number of samples (i.e. episodes in batch)
        elif hparams.model_type == 'NoisyPG':
            ...
        else:
            raise Exception('Unsupported model type.')
        
        loss.backward()
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
        
    return best_score, best_episode


#############   Main

model = models.PGNetwork(inputSize=GRID_SIZE,
                        hiddenSize=hparams.hidden,
                        outputSize=OUTPUT,
                        leaky=hparams.leaky,
                        temperature=hparams.temperature,
                        nHiddenLyrs=hparams.num_hiddens).to(DEVICE)
best_score, best_episode = train(hparams,model)
print(f'\ntraining completed\nbest score: {best_score} achieved at episode {best_episode}')

with open(PATH+'/digest.txt','w') as f:
    f.write(f'best episode:{best_episode}\nbest score: {best_score}')