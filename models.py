import torch
import torch.nn.functional as F
import numpy as np
from math import floor


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

class CNN_PG(torch.nn.Module):
    def __init__(self,hparams,w,h,outputSize):
        super(CNN_PG,self).__init__()
        self.leaky = hparams.leaky
        self.sigmoid = hparams.sigmoid
        self.num_layers = hparams.num_hiddens
        self.hiddenSize = hparams.hidden
        self.outputSize = outputSize

        #layer definitions
        self.temperature = torch.tensor(hparams.temperature if hparams.temperature > 0 else 1e-8, dtype=float)
        self.activations = {False:F.relu, True:F.leaky_relu}    #allows user to specify hidden activations
        
        self.conv1 = torch.nn.Conv2d(in_channels=2,out_channels=3, kernel_size=7, padding=2, stride=1, dilation=1)
        w,h = self.__outSize(w,7,padLength=2), self.__outSize(h,7,padLength=2)  #output of conv1 is out_channels x w x h

        self.pool = torch.nn.MaxPool2d(kernel_size=2,padding=0,stride=1,dilation=1)
        w,h = self.__outSize(w,2,padLength=0), self.__outSize(h,2,padLength=0)  #output of pool is out_channels x w x h
        
        self.conv2 = torch.nn.Conv2d(in_channels=3,out_channels=6, kernel_size=5, padding=2, stride=1, dilation=1)
        w,h = self.__outSize(w,5,padLength=2), self.__outSize(h,5,padLength=2)  #output of conv2 is out_channels x w x h

        self.linear1 = torch.nn.Linear(6*w*h, 100)  #linear layer takes a 1D tensor length out_channels * w * h, requires flattened tensor
        self.linear2 = torch.nn.Linear(100,50)      
        self.linear3 = torch.nn.Linear(50,outputSize)

    def forward(self,x):
        x = self.activations[self.leaky](self.conv1(x))
        x = self.pool(x)
        x = self.activations[self.leaky](self.conv2(x))
        x = self.activations[self.leaky](self.linear1(torch.flatten(x)))
        x = self.activations[self.leaky](self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x
        
    def __outSize(self, inputSize, kSize, padLength=0, stride=1, dilation=1):
        return floor( ((inputSize + 2*padLength - dilation * (kSize - 1) - 1) / stride) + 1 )