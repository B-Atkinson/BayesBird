import torch
import torch.nn.functional as F
import numpy as np
import math


class PGNetwork(torch.nn.Module):
    '''A neural network to produce a probability of going up.'''
    def __init__(self, hparams, inputSize, outputSize):
        super(PGNetwork,self).__init__()
        #define layers for probability
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