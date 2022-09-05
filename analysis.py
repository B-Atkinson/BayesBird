#!/usr/bin/env python3
from cProfile import run
import csv
from email import parser
import pathlib
import os
from turtle import color
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
import multiprocessing as mp
import os
import argparse
# import cupy as cp         #this needs to be uncommented if processing data created using gpus

def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run analysis for FlappyBird reinforcement learning with human influence.')    
    parser.add_argument('--rootDirectory', type=str, default='/home/brian.atkinson/thesis/data',
                        help='The root of the experiment tree containing all experimental directories to be analyzed. The default is that ')        
    return parser.parse_args()

def isParent(dir):
    '''The name isParent is slightly misleading, as it determines if the inputted 
       directory is the parent to the individual test folders. Each test is con-
       sidered a child in this function.'''
    path = pathlib.Path(dir) if not isinstance(dir,pathlib.PosixPath) else dir
    suffixes = ['.csv','.png','.txt','.p']
    for folder in path.iterdir():
        for subfolder in folder.iterdir():
            suf = subfolder.suffix
            if suf in suffixes: 
                return True
    return False

def isTest(dir):
    '''Determines if the directory is a test based on file types contained.'''
    path = pathlib.Path(dir) if not isinstance(dir,pathlib.PosixPath) else dir
    suffixes = ['.csv','.png','.txt','.p']
    for folder in path.iterdir():
        suf = folder.suffix
        if suf in suffixes: 
            return True
    return False

def getChildren(dir):
    '''Returns a list of pathlib.PosixPaths to the children of the inputted directory.'''
    path = pathlib.Path(dir) if not isinstance(dir,pathlib.PosixPath) else dir
    return list(child for child in path.iterdir())

def DFS_tree(dirPath):
    '''Searches whatever directory is inputted to find all the individual tests done down the tree.
       Returns a list of pathlib.PosixPaths to each test across all experiments. Implements Depth-
       First Search to find all tests.'''
    dir = pathlib.Path(dirPath) if not isinstance(dirPath,pathlib.PosixPath) else dirPath
    if isTest(dir):
        #base case where the directory is a test, return itself
        return [dir]
    results = []
    for subfolder in dir.iterdir():
        if isParent(subfolder):
            #current directory is the grandparent of the tests
            results.extend(getChildren(subfolder))
        else:
            #tests are more than 2 levels down, keep searching
            results.extend(DFS_tree(subfolder))
    return results

# you have to use str2bool
# because of an issue with argparser and bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def rearrange(oldGame):
    frames,neurs = np.shape(oldGame)
    neurons = np.zeros((neurs,frames))
    for frame in range(frames):
        actList = list(oldGame[frame])
        for neuron in range(neurs):
            neurons[neuron, frame] = actList[neuron]
    return neurons


def main(path):
    dataDir = pathlib.Path(path) if not isinstance(path,pathlib.PosixPath) else path
    
    #if reach here, dataDir is the parent directory of the tests
    batch = 20
    tests = [dataDir]    

    for dir in tests:  
        sum = 0
        try:  
            experiment = os.path.split(dir)[1]
            print('opening:',dir)
            eps=[]
            raw_scores=[]

            keyDict = {"seed":'Seed',
            "L2":'L2',
            "leaky":'Activation',
            "human":'Human Agent'
            }
            expDict = {}
            with open(os.path.join(dir,'metadata.txt'),'r') as fd:
                lines = fd.readlines()[1:-1]
            
            for line in lines:
                parts = line.split(':')
                rawKey=parts[0][3:-1]
                if rawKey in keyDict.keys():
                    key = keyDict[rawKey]
                    value = parts[1].split(',')[0][1:]
                    value = value.lstrip('"')
                    value = value.strip('"')
                    expDict[key] = value

        except Exception as e:
            print('\n*** {} caught error {}***\n'.format(dir,sum),flush=True)

        cumulative_scores = []
        running_rewards = []
        r_sum = 0   
        running_reward = 0
        maximum = 0 
        checkpoints = []
        points = []

        for e in range(len(eps)):
            
            if e % batch == 0:
                #treat batch # of games as an episode for plotting and calculating running reward
                cumulative_scores.append(r_sum)
                running_reward = .99*running_reward+.01*r_sum/batch
                running_rewards.append(running_reward)
                r_sum = 0
                
                if running_rewards[-1] >= 4:
                    print('\n***file:{} file length disagreement***'.format(dir))
                if len(running_rewards)>=2 and running_rewards[-1]<running_rewards[-2]:
                    minimum = len(running_rewards)
                if running_rewards[-1] > running_rewards[maximum]:
                    maximum = len(running_rewards)

            if (e % splits == 0):
                checkpoints.append(running_reward)
                points.append(e)


            r_sum += raw_scores[e]
        
        with open(os.path.join(dir,'digest.txt'),'w') as f:
            f.write('directory:{}\n'.format(experiment))
            f.write('best game:{}, best score:{}\n'.format(bestGame,bestScore))
            f.write('max score at epoch:{}, game:{}, reward:{}\n'.format(maximum,maximum*20,running_rewards[maximum]))
            f.write('min score at epoch:{}, game:{}\n\n'.format(minimum, minimum*20))
            f.write('running reward checkpoints\n')
            
            for e in points:
                f.write('{} & '.format(e))
            f.write('\n')
            for rr in checkpoints:
                f.write('{:.5f} & '.format(rr))     

            
        length = len(cumulative_scores)
        plt.clf()
        plt.scatter(range(length),cumulative_scores,marker='.')
        plt.title('Episode Scores ({})'.format(title))
        plt.xlabel('Episodes (Batch of {} games)'.format(batch))
        plt.ylabel('Number of pipes')
        plt.savefig(os.path.join(dir,'num_pipes.png'))
        plt.clf()

        plt.scatter(range(length),running_rewards,marker='.')
        plt.title('Running Reward ({})'.format(title))
        plt.xlabel('Episodes (Batch of {} games)'.format(batch))
        plt.ylabel('Running Average Score')
        plt.savefig(os.path.join(dir,'running_reward.png'))
        plt.clf()

        print('done analyzing',dir,flush=True)
        
        return

if __name__=='__main__':
    '''Leverages multi-processing to be able to conduct graphical and numerical
       analysis on an arbitrary number of tests nearly simultaneously. This script will 
       first locate each individual test in the data directory tree, then create a process 
       to analyze the tests with a 1:1 process to test ratio.'''
    params = make_argparser()
    data = pathlib.Path(params.rootDirectory)
    tgtList = DFS_tree(data)
    pool = mp.Pool(len(tgtList))
    try:
        results = pool.map_async(main,tgtList)  
    except Exception as e:
        print(e)
    finally:
        pool.close()
        pool.join()
        print('\n***all processes of {} are done***'.format(str(data)))