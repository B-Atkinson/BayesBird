import torch
import torch.nn.functional as F
import numpy as np
import math

class PGNetwork(torch.nn.Module):
    '''A neural network to produce a probability of going up.'''
    def __init__(self, inputSize, hiddenSize, outputSize, leaky=False, sigmoid=False, temperature=1,nHiddenLyrs=2):
        super(PGNetwork,self).__init__()
        #define layers for probability
        self.layers = {1 : torch.nn.Linear(inputSize, hiddenSize)}
        if nHiddenLyrs <= 1:
            self.layers[2] = torch.nn.Linear(hiddenSize, outputSize)
        else:
            for lyr in range(2,nHiddenLyrs+1):
                self.layers[lyr] = torch.nn.Linear(hiddenSize, hiddenSize)
            self.layers[nHiddenLyrs+1] = torch.nn.Linear(hiddenSize, outputSize)
        
        self.leaky = leaky
        self.sigmoid = sigmoid
        self.num_layers = nHiddenLyrs+1
        self.temperature = torch.tensor(temperature if temperature > 0 else 1e-6, dtype=float)
        self.activations = {False:F.relu, True:F.leaky_relu}
        
    def forward(self,x):
        #network for probability
        for lyr in range(self.num_layers):
            x = self.activations[self.leaky](self.layers[lyr](x))
        
        p = self.layers[self.num_layers](x)
        if self.sigmoid:
            p = torch.sigmoid(p / self.temperature)
        return p
        
class NoisyPG(torch.nn.Module):
    '''A neural network that uses heteroscedastic uncertainty to produce a probability of going up. uses l2 normalization.'''
    def __init__(self, inputSize, hiddenSize, outputSize, leaky=False,p_keep=.5):
        super(NoisyPG,self).__init__()
        self.L1 = torch.nn.Linear(inputSize, hiddenSize)
        self.L2 = torch.nn.Linear(hiddenSize, hiddenSize)
        self.L3 = torch.nn.Linear(hiddenSize, outputSize)
        
        self.pkeep = p_keep
        self.leaky = leaky
        self.activations = {False:F.relu, True:F.leaky_relu}
        
    def forward(self,x):
        #network for probability
        l1_out = self.activations[self.leaky](self.L1(x))
        l2_out = self.activations[self.leaky](self.L2(l1_out))
        p = torch.sigmoid(self.L3(l2_out))
        return p