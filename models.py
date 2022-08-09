import torch
import torch.nn.functional as F
import numpy as np
import math


class PGNetwork(torch.nn.Module):
    '''A policy gradient neural network to produce a probability of going up. Can  be  instantiated  with  an 
    arbitrary number of linear hidden layers, with a minimum number of 1. Can also use leaky or  conventional
    relu  activation  in the hidden layers, as well as linear or sigmoid activation  in the final layer. This
    network  requires inputs to be 1D, meaning a 2D frame must be flattened before input. 

    Input:
    hparams- an argparse parameter object containing runtime parameters
    inputSize- the 1D size of inputs to the network
    outputSize- number of outputs expected from the network, defaults to 1
    '''
    def __init__(self, hparams, inputSize, outputSize=1):
        super(PGNetwork,self).__init__()
        #class attributes
        self.leaky = hparams.leaky
        self.sigmoid = hparams.sigmoid
        self.num_layers = hparams.num_hiddens
        self.hiddenSize = hparams.hidden
        self.outputSize = outputSize
        self.softmax = hparams.softmax
        self.temperature = torch.tensor(hparams.temperature if hparams.temperature > 0 else 1e-8, dtype=float)
        self.activations = {False:F.relu, True:F.leaky_relu}    #allows user to specify hidden activations
        self.layers = torch.nn.ModuleList()                     #this will store the layers of the network

        self.layers.append( torch.nn.Linear(inputSize, self.hiddenSize) )
        if self.num_layers <= 1:
            self.layers.append( torch.nn.Linear(self.hiddenSize, outputSize) )
        else:
            for lyr in range(2,self.num_layers+1):
                self.layers.append( torch.nn.Linear(self.hiddenSize, self.hiddenSize) )
            self.layers.append( torch.nn.Linear(self.hiddenSize, outputSize) )         
        print(len(self.layers),flush=True)

    def forward(self,x):
        '''Takes a 1D input vector and outputs a probability.'''
        #network for probability
        for lyr in range(self.num_layers):
            x = self.activations[self.leaky](self.layers[lyr](x))
        
        p = self.layers[self.num_layers](x)
        if self.sigmoid:
            p = torch.sigmoid(p / self.temperature)
        elif self.softmax:
            if self.outputSize > 1:
                p = F.softmax(p)
            else:
                raise Exception('Softmax requires output size >1 to be useful')
        return p
        
class NoisyPG(torch.nn.Module):
    '''A neural network that uses heteroscedastic uncertainty to produce a probability of going up. uses l2 normalization.'''
    def __init__(self, hparams, inputSize, outputSize):
        super(NoisyPG,self).__init__()
        #define layers for probability
        raise NotImplementedError('This model has not been developed yet.')

        self.leaky = hparams.leaky
        self.sigmoid = hparams.sigmoid
        self.num_layers = hparams.num_hiddens
        self.hiddenSize = hparams.hidden
        self.temperature = torch.tensor(hparams.temperature if hparams.temperature > 0 else 1e-6, dtype=float)
        self.activations = {False:F.relu, True:F.leaky_relu}

        self.layers = torch.nn.ModuleList()
        self.layers.append( torch.nn.Linear(inputSize, self.hiddenSize) )
        if self.num_layers <= 1:
            self.layers.append( torch.nn.Linear(self.hiddenSize, outputSize) )
        else:
            for lyr in range(2,self.num_layers+1):
                self.layers.append( torch.nn.Linear(self.hiddenSize, self.hiddenSize) )
            self.layers.append( torch.nn.Linear(self.hiddenSize, outputSize) )  
        print(len(self.layers),flush=True)
        
    def forward(self,x):
        #network for probability
        for lyr in range(self.num_layers):
            x = self.activations[self.leaky](self.layers[lyr](x))
        
        p = self.layers[self.num_layers](x)
        if self.sigmoid:
            p = torch.sigmoid(p / self.temperature)
        return p