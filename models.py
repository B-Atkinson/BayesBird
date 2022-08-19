from turtle import forward
import torch
import torch.nn.functional as F
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
    def __init__(self, hparams, inputSize, outputSize, DEVICE):
        super(PGNetwork,self).__init__()
        #class attributes
        self.leaky = hparams.leaky
        self.sigmoid = hparams.sigmoid
        self.hiddenSize = hparams.hidden
        self.outputSize = outputSize
        self.softmax = hparams.softmax
        self.DEVICE = DEVICE
        self.dropout_layer = GaussianDropout if 'GAUSS' in hparams.dropout_type.upper() else BernoulliDropout
        self.temperature = torch.tensor(hparams.temperature if hparams.temperature > 0 else 1e-8, dtype=float)
        self.activations = {False:F.relu, True:F.leaky_relu}    #allows user to specify hidden activations
        self.layers = torch.nn.ModuleList()                     #this will store the layers of the network

        self.layers.append( torch.nn.Linear(inputSize, self.hiddenSize) )
        self.layers.append( self.dropout_layer(hparams.dropout,hparams.seed,DEVICE) )

        if hparams.num_hiddens <= 1:
            self.layers.append( torch.nn.Linear(self.hiddenSize, outputSize) )
        else:
            for lyr in range(2,hparams.num_hiddens+1):
                self.layers.append( torch.nn.Linear(self.hiddenSize, self.hiddenSize) )
                self.layers.append( self.dropout_layer(hparams.dropout,hparams.seed,DEVICE) )
            self.layers.append( torch.nn.Linear(self.hiddenSize, outputSize) )

        self.num_layers = len(self.layers)
        print(self.num_layers,flush=True)  #should be twice the number of specified hidden layers due to dropout

    def forward(self,x):
        '''Takes a 1D input vector and outputs a probability.'''
        #network for probability
        for lyr in range(0,self.num_layers-1,2):
            x = self.layers[lyr+1](self.activations[self.leaky](self.layers[lyr](x)))
        
        p = self.layers[self.num_layers-1](x)
        if self.sigmoid:
            p = torch.sigmoid(p / self.temperature)
        elif self.softmax:
            if self.outputSize > 1:
                p = F.softmax(p)
            else:
                raise Exception('Softmax requires output size >1 to be useful')
        return p

    def evaluate(self,x,m=10):
        '''After training using m=1 for Monte Carlo, use a larger m-value to get more accurate result. 
        Default m-value is 10.'''
        inferences = torch.zeros(m).to(self.DEVICE)
        for i in range(m):
            inferences[i] = self.forward(x).detach()
        return torch.mean(inferences)
        
class CNN_PG(torch.nn.Module):
    '''Uses a deep CNN to implement a policy gradient network. Compares current state'''
    def __init__(self,hparams,w,h,outputSize,DEVICE):
        super(CNN_PG,self).__init__()
        self.leaky = hparams.leaky
        self.sigmoid = hparams.sigmoid
        self.num_layers = hparams.num_hiddens
        self.hiddenSize = hparams.hidden
        self.outputSize = outputSize
        self.DEVICE = DEVICE
        self.dropout_layer = GaussianDropout if 'GAUSS' in hparams.dropout_type.upper() else BernoulliDropout

        #layer definitions
        self.temperature = torch.tensor(hparams.temperature if hparams.temperature > 0 else 1e-8, dtype=float)
        self.activations = {False:F.relu, True:F.leaky_relu}    #allows user to specify hidden activations
        
        ch1,kSize1,pad1,stride1 = 32,8,3,4
        self.conv1 = torch.nn.Conv2d(in_channels=2,out_channels=ch1, kernel_size=kSize1, padding=pad1, stride=stride1)
        w,h = self.__outSize(w,kSize=kSize1,padLength=pad1,stride=stride1), self.__outSize(h,kSize=kSize1,padLength=pad1,stride=stride1)  #output of conv1 is ch1 x w x h
        self.d1 = self.dropout_layer(hparams.dropout,hparams.seed,DEVICE)

        ch2,kSize2,pad2,stride2 = 64,4,2,1
        self.conv2 = torch.nn.Conv2d(in_channels=ch1,out_channels=ch2, kernel_size=kSize2, padding=pad2, stride=stride2)
        w,h = self.__outSize(w,kSize=kSize2,padLength=pad2,stride=stride2), self.__outSize(h,kSize=kSize2,padLength=pad2,stride=stride2)  #output of pool is ch1 x w x h
        self.d2 = self.dropout_layer(hparams.dropout,hparams.seed,DEVICE)
        
        ch3,kSize3,pad3,stride3 = 64,3,0,1
        self.conv3 = torch.nn.Conv2d(in_channels=ch2,out_channels=ch3, kernel_size=kSize3, padding=pad3, stride=stride3)
        w,h = self.__outSize(w,kSize=kSize3,padLength=pad3,stride=stride3), self.__outSize(h,kSize=kSize3,padLength=pad3,stride=stride3)  #output of pool is ch1 x w x h
        self.d3 = self.dropout_layer(hparams.dropout,hparams.seed,DEVICE)

        self.linear1 = torch.nn.Linear(ch3*w*h, 200)  #linear layer takes a 1D tensor length out_channels * w * h, requires flattened tensor
        self.d4 = self.dropout_layer(hparams.dropout,hparams.seed,DEVICE)
        
        self.linear2 = torch.nn.Linear(200,50)      
        self.d5 = self.dropout_layer(hparams.dropout,hparams.seed,DEVICE)
        
        self.linear3 = torch.nn.Linear(50,outputSize)

    def forward(self,x):
        #utilize dropout during training
        x = self.d1(self.activations[self.leaky](self.conv1(x)))
        x = self.d2(self.activations[self.leaky](self.conv2(x)))
        x = self.d3(self.activations[self.leaky](self.conv3(x)))
        x = self.d4(self.activations[self.leaky](self.linear1(torch.flatten(x))))
        x = self.d5(self.activations[self.leaky](self.linear2(x)))
        if self.sigmoid:
            x = torch.sigmoid(self.linear3(x))
        else:
            x = self.activations[self.leaky](self.linear3(x))
        return x
            

    def evaluate(self,x,m=10):
        '''After training using m=1 for Monte Carlo, use a larger m-value to get more accurate result. 
        Default m-value is 10.'''
        inferences = torch.zeros(m).to(self.DEVICE)
        for i in range(m):
            inferences[i] = self.forward(x).detach()
        return torch.mean(inferences)

        
    def __outSize(self, inputSize, kSize, padLength=0, stride=1, dilation=1):
        return floor( ((inputSize + 2*padLength - dilation * (kSize - 1) - 1) / stride) + 1 )

class GaussianDropout(torch.nn.Module):
    '''Applies noise to every element of the input tensor from sampling a mean=1, stddev=p/(1-p) Normal
    distribution. 
    References:
    https://discuss.pytorch.org/t/gaussiandropout-implementation/151756/4
    https://stackoverflow.com/questions/65501878/gaussiandropout-vs-dropout-vs-gaussiannoise-in-keras
    
    Input:
    p_drop- the probability of an element in the input tensor to be zeroed out
    seed- an integer that is used to seed the prng'''
    def __init__(self,p_drop=.5,seed=1,DEVICE=None):
        super(GaussianDropout,self).__init__()
        self.DEVICE = DEVICE
        self.alpha = p_drop / (1 - p_drop)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(self, x):
        '''Applies noise to every element of the input tensor from sampling a mean=1, stddev=p/(1-p) Normal
        distribution.
        
        Input:
        x- a tensor of activations/values to apply dropout to
        
        Output:
        x- the tensor result of multiplicative Gaussian noise being applied to the input tensor'''
        noise = torch.normal(mean=1,std=self.alpha,generator=self.generator,size=x.size()).to(self.DEVICE)
        x = torch.mul(x,noise)
        return x

class BernoulliDropout(torch.nn.Module):
    '''Creates a dropout layer using a Bernoulli distribution with a probability of sampling 1
    defined as (1 - p_drop). 
    
    Input:
    p_drop- the probability of an element in the input tensor to be zeroed out'''
    def __init__(self,p_drop=.5,seed=1,DEVICE=None):
        super(BernoulliDropout,self).__init__()
        self.DEVICE = DEVICE
        self.p_drop = p_drop
        self.bern = torch.distributions.bernoulli.Bernoulli(probs=(1-p_drop))
    
    def forward(self,x):
        '''Applies noise to every element of the input tensor from sampling a Bernoulli distribution.
        
        Input:
        x- a tensor of activations/values to apply dropout to
        
        Output:
        x- the tensor result of Bernoulli noise being applied to the input tensor'''
        noise = self.bern.sample(sample_shape=x.size()).to(self.DEVICE)
        x = torch.mul(x,noise)
        return x
        