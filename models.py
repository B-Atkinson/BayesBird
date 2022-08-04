import torch
import torch.nn.functional as F
import numpy as np
import math

class PGNetwork(torch.nn.Module):
    '''A neural network to produce a probability of going up.'''
    def __init__(self, inputSize, hiddenSize, outputSize, leaky=False, sigmoid=False, temperature=1):
        super(PGNetwork,self).__init__()
        #define layers for probability
        self.L1 = torch.nn.Linear(inputSize, hiddenSize)
        self.L2 = torch.nn.Linear(hiddenSize, hiddenSize)
        self.L3 = torch.nn.Linear(hiddenSize, outputSize)
        
        self.leaky = leaky
        self.sigmoid = sigmoid
        self.temperature = torch.tensor(temperature if temperature > 0 else 1e-6, dtype=float)
        self.activations = {False:F.relu, True:F.leaky_relu}
        
    def forward(self,x):
        #network for probability
        l1_out = self.activations[self.leaky](self.L1(x))
        l2_out = self.activations[self.leaky](self.L2(l1_out))
        p = self.L3(l2_out)
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