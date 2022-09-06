from textwrap import indent
import torch
import os
import pathlib
import json
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import pygame
from pygame.constants import K_w
import params
import utils
import models
from matplotlib import pyplot as plt
import matplotlib as mpl ; mpl.use("Agg")
import matplotlib.animation as manimation

dir = '/home/brian.atkinson/Bayes/data/ShallowTest/PGNetwork'
# hparams = params.get_hparams()

# #check if a CUDA GPU is available
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LOADER_KWARGS = {'num_workers': hparams.workers, 'pin_memory': True} if torch.cuda.is_available() else {}

# #if no CUDA GPU available, check for Apple M1-specific GPU called MPS
# if not torch.cuda.is_available():
#     DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     if torch.backends.mps.is_available():
#         gpu='MPS'
#     else:
#         gpu=False
# else:
#     if torch.cuda.is_available():
#         gpu='CUDA'
#     else:
#         gpu=False

# print('loading params',flush=True)

# #load parameter dictionary with values from experiment
# with open(os.path.join(dir,'metadata.txt'),'r') as fp:
#     lines = fp.readlines()
# string = ''.join(lines)
# hparams.__dict__ = json.loads(string)

# print('params loaded',flush=True)

# #specified in ple/__init__.py lines 187-194
# WIDTH = hparams.screenWidth     #if downsample by 4, w=72
# HEIGHT = hparams.screenHeight    #if downsample by 4, h=100
# GRID_SIZE = WIDTH * HEIGHT
# GAP = 150
# OUTPUT = 1
# ACTION_MAP = {'flap': K_w,'noop': None}
# REWARDDICT = {"positive":1, "loss":-1}

# #if environment does not support image rendering, this turns it off
# if not hparams.render:
#     os.environ['SDL_VIDEODRIVER'] = 'dummy'

# #initialize a model then load with trained parameters
# if 'PGN' in hparams.model_type.upper():
#     model = models.PGNetwork(hparams,GRID_SIZE,OUTPUT,DEVICE).to(DEVICE)
# elif 'CNN' in hparams.model_type.upper():
#     model = models.CNN_PG(hparams, w=WIDTH, h=HEIGHT, outputSize=OUTPUT,DEVICE=DEVICE).to(DEVICE)
# else:
#     raise Exception('Unsupported model type.')

# try:
#     model.load_state_dict(torch.load(os.path.join(dir,'best_model.pt')))
# except RuntimeError:
#     model.load_state_dict(torch.load(os.path.join(dir,'best_model.pt'),map_location=torch.device('cpu')))

# print('model loaded',flush=True)

# #load a game environment
# FLAPPYBIRD = FlappyBird(pipe_gap=GAP, rngSeed=hparams.seed)
# game = PLE(FLAPPYBIRD, display_screen=hparams.render, force_fps=True, rng=hparams.seed, reward_values=REWARDDICT)
# game.init() 

# print('simulating play',flush=True)

# high_score = -1
# while high_score < 10:
#     game.reset_game()
#     num_pipes = 0
#     prettyFrames = []

#     #play a single game
#     while not game.game_over():
#         #retrieve current game state and process into tensor
#         prettyFrames.append( game.getScreenRGB() )
#         currentFrame = game.getScreenGrayscale()
#         frame_np = currentFrame[:,:400]
#         frame_np = frame_np[::4,::4]
#         norm_np = (frame_np-frame_np.mean())/(frame_np.std()+np.finfo(float).eps)
        
#         #choose the appropriate model and get our action
#         if 'PGN' in hparams.model_type.upper():
#             combined_np = norm_np.ravel()
#             # plt.imshow(combined_np.reshape(WIDTH,HEIGHT))
#             frame_t = torch.from_numpy(combined_np).float().to(DEVICE)     #convert to a tensor
#             p = model(frame_t)
#         elif 'CNN' in hparams.model_type.upper():
#             # frame_stack = np.stack([norm_np,lastFrame],0)
#             # processed = np.concatenate([utils.processScreen(frame_stack,frame_mean,frame_std),np.zeros((WIDTH,HEIGHT,1))],2)
#             # plt.imshow(processed)
#             frame_t = torch.from_numpy(norm_np.reshape([1,WIDTH,HEIGHT])).float().to(DEVICE)
#             p = model(frame_t)
#         else:
#             raise Exception('Unsupported model type.')         

#         #get the action to take
#         if hparams.dropout == 0.:
#             dist = torch.distributions.bernoulli.Bernoulli(p)
#             action = dist.sample()
#         else:
#             action = 1 if p >= .5 else 0
        
#         #take the action and record data for this step
#         reward = game.act(ACTION_MAP['flap'] if action==1 else ACTION_MAP['noop'])
#         if reward > 0:
#             num_pipes += 1
#     #end of game play for episode
#     if num_pipes > high_score:
#         high_score = num_pipes
#         print(f'high score: {num_pipes}',flush=True)

# print('done simulating play')

#save numpy arrays to png for later flipping
global movieFrameDir
movieFrameDir = os.path.join(dir,f'movieFrames')
# os.makedirs(movieFrameDir,exist_ok=False)
# for i in range(len(prettyFrames)):
#     plt.imsave(os.path.join(movieFrameDir,f'frame{i}.png'),prettyFrames[i])

# print('sideways frames saved',flush=True)

#load images and then save them with proper orientation
numFrames = len(list(f for f in pathlib.Path(movieFrameDir).iterdir()))
# for i in range(numFrames):
#     framePath = os.path.join(movieFrameDir,f'frame{i}.png')
#     frame = pygame.image.load(framePath)
#     rotatedFrame = pygame.transform.rotate(frame,-90)
#     pygame.image.save(rotatedFrame,framePath)

# print('done rotating frames',flush=True)

print('making movie',flush=True)

movie_title = 'demo.mp4'
resolution = 75

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title=movie_title, artist='atkinson', comment='Bayes-Bird')
writer = FFMpegWriter(fps=8, metadata=metadata)

f = plt.figure(figsize=[6, 6*1.39], dpi=resolution)



f = plt.figure(figsize=[6, 6*1.39], dpi=resolution)
# f = plt.figure(dpi=resolution)
with writer.saving(f,dir + f'/{movie_title}', resolution):
    for i in range(numFrames):
        frame = plt.imread(movieFrameDir+f'/frame{i}.png').astype('uint8')
        plt.imshow(frame)
        print('no errors here')
        writer.grab_frame()
        f.clear()



print('finished making movie',flush=True)
