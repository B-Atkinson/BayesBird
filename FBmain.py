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
import cv2



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

#specified in ple/__init__.py lines 187-194
WIDTH = hparams.screenWidth     #if downsample by 4, w=72
HEIGHT = hparams.screenHeight    #if downsample by 4, h=100
GRID_SIZE = WIDTH * HEIGHT
GAP = 150
ACTION_MAP = {'flap': K_w,'noop': None}
REWARDDICT = {"positive":1, "loss":-1}
OUTPUT = 2 if hparams.softmax else 1
PATH = hparams.output_dir + hparams.model_type
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

    # frame_mean, frame_std = utils.frameBurnIn(game,hparams,model,WIDTH,HEIGHT,DEVICE,gpu)

    training_summaries = []
    best_score, best_episode = -1,0
    
    with open(os.path.join(PATH,'output.txt'),'a') as f:
        f.write(f'commencing training with {hparams.model_type} model\n')
    print(f'commencing training with {hparams.model_type} model',flush=True)
    
    #train for num_episodes
    for episode in range(1,1+hparams.num_episodes):
        game.reset_game()
        
        agent_score, num_pipes = 0, 0
        frames, actions, rewards, probs, log_ps = [], [], [], [], []
        
        #play a single game
        while not game.game_over():
            #retrieve current game state and process into tensor
            currentFrame = game.getScreenGrayscale()
            frame_np = currentFrame[:,:400]
            frame_np = frame_np[::4,::4]
            norm_np = (frame_np-frame_np.mean())/(frame_np.std()+np.finfo(float).eps)
            
            #choose the appropriate model and get our action
            if 'PGN' in hparams.model_type.upper():
                combined_np = norm_np.ravel()
                frame_t = torch.from_numpy(combined_np).float().to(DEVICE)     #convert to a tensor
                p = model(frame_t)
            elif 'CNN' in hparams.model_type.upper():
                frame_t = torch.from_numpy(norm_np.reshape([1,WIDTH,HEIGHT])).float().to(DEVICE)
                p = model(frame_t)
            else:
                raise Exception('Unsupported model type.')         

            #get human action augmentation with potentially decayed human influence
            if hparams.human_influence > 0.:
                hInfluence = (hparams.human_influence * (hparams.human_decay ** episode)) if hparams.human_decay else hparams.human_influence
                state = game.getGameState()
                if state['next_pipe_dist_to_player']<=80:
                    if state['player_y'] >(state['next_pipe_bottom_y']-32):
                        #flap if player is in line or below the bottom edge of the gap
                        p += hInfluence
                        if p > 1:
                            #ensure probability never goes above 1 without destroying gradient info
                            p -= (p-1)
                    else:
                        #otherwise enforce going down
                        p -= hInfluence
                        if 0 > p:
                            #ensure probability never dips below 0 without destroying gradient info
                            p -= p
            
            #get the action to take
            if hparams.dropout == 0.:
                dist = torch.distributions.bernoulli.Bernoulli(p)
                action = dist.sample()
                logp = dist.log_prob(action)
                log_ps.append(logp)
            else:
                action = 1 if p >= .5 else 0
                logp = torch.log( p if action==1 else 1-p  )
                log_ps.append(logp)
            
            #take the action
            reward = game.act(ACTION_MAP['flap'] if action==1 else ACTION_MAP['noop'])
            agent_score += reward
            
            #record data for this step
            if reward > 0:
                num_pipes += 1
            frames.append(currentFrame)
            actions.append(1 if action==K_w else 0) #flaps stored as 1, no-flap stored as 0
            rewards.append(reward)
            probs.append(p)
        #end of game play for episode
            
        #update performance variables
        if num_pipes > best_score:
            string = ''
            best_score = num_pipes
            best_episode = episode
            with open(os.path.join(PATH,'output.txt'),'a') as f:
                f.write(f'\n{string}new high score:{best_score} episode:{best_episode}\n')
            torch.save(model.state_dict(), os.path.join(PATH,'best_model.pt'),pickle_protocol=4)
        training_summaries.append( (episode, num_pipes) )
        stacked_rewards = np.vstack(rewards)
        discounted_reward, num_sets = utils.discount_rewards(stacked_rewards, hparams.gamma, hparams.discount_vector)
        discounted_reward = torch.tensor(discounted_reward).float().to(DEVICE)
        num_sets =  torch.tensor(num_sets).float().to(DEVICE)
        
        #calculate loss and do backprop
        logp = torch.stack(log_ps)            #add all log probabilities in the episode
        loss = torch.mul(discounted_reward,logp)
        loss = torch.div(loss, hparams.batch_size) #divide by number of samples (i.e. episodes in batch)
        
        
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
        
    agent_score,num_pipes = 0,0
    frames = []
    f = 0    
    #play a single game
    while not game.game_over():
        f+=1
        #retrieve current game state in RGB for movie-making
        prettyFrame = game.getScreenRGB()
        plt.imsave(os.path.join(FRAMES,f'evaluate_{f}.png'),prettyFrame)
        
        #retrieve current game state and process into tensor
        currentFrame = game.getScreenGrayscale()
        frame_np = currentFrame[:,:400]
        frame_np = frame_np[::4,::4]
        norm_np = (frame_np-frame_np.mean())/(frame_np.std()+np.finfo(float).eps)
        
        #choose the appropriate model and get our action
        if 'PGN' in hparams.model_type.upper():
            combined_np = norm_np.ravel()
            frame_t = torch.from_numpy(combined_np).float().to(DEVICE)     #convert to a tensor
            p = model.evaluate(frame_t) if hparams.dropout != 0. else model(frame_t)
        elif 'CNN' in hparams.model_type.upper():
            frame_t = torch.from_numpy(norm_np.reshape([1,WIDTH,HEIGHT])).float().to(DEVICE)
            p = model.evaluate(frame_t) if hparams.dropout != 0. else model(frame_t)
        else:
            raise Exception('Unsupported model type.')     

        #get the action to take
        action = 1 if p >= .5 else 0
        
        #take the action
        reward = game.act(ACTION_MAP['flap'] if action==1 else ACTION_MAP['noop'])
        agent_score += reward
            
        #record data for this step
        if reward > 0:
            num_pipes += 1
            frames.append(currentFrame)
    return num_pipes

    

#############   Main
FLAPPYBIRD = FlappyBird(pipe_gap=GAP, rngSeed=hparams.seed)
game = PLE(FLAPPYBIRD, display_screen=False, force_fps=True, rng=hparams.seed, reward_values=REWARDDICT)


#train a model
if 'PGN' in hparams.model_type.upper():
    model = models.PGNetwork(hparams,GRID_SIZE,OUTPUT,DEVICE).to(DEVICE)
elif 'CNN' in hparams.model_type.upper():
    model = models.CNN_PG(hparams, w=WIDTH, h=HEIGHT, outputSize=OUTPUT,DEVICE=DEVICE).to(DEVICE)
else:
    raise Exception('Unsupported model type.')  

#evaulate the model resulting from full training length
lastModel, (best_score, best_episode) = train(hparams,model,game)
with open(os.path.join(PATH,'output.txt'),'a') as f:
    f.write(f'\ntraining completed\nbest score: {best_score} achieved at episode {best_episode}\n\nbeginning evaluation\n')
print(f'\ntraining completed\nbest score: {best_score} achieved at episode {best_episode}\n\nbeginning evaluation\n',flush=True)

#attempt to evaluate the best overall model from training
try:
    model.load_state_dict(torch.load(os.path.join(PATH,'best_model.pt')))
    game = FLAPPYBIRD = FlappyBird(pipe_gap=GAP, rngSeed=hparams.seed+10)
    game = PLE(FLAPPYBIRD, display_screen=hparams.render, force_fps=not hparams.render, rng=hparams.seed+10, reward_values=REWARDDICT)
    num_pipes = evaluate(hparams,model,game)
    with open(os.path.join(PATH,'output.txt'),'a') as f:
        f.write(f'\nevaluation completed\nscore: {num_pipes}\n')
except Exception as e:
    print(e)
    with open(PATH+'/output.txt','w') as f:
        f.write(f'Unable to load best model\n')

